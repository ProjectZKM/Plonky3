//! whir-gc experiment: closing the shared sumcheck by a column sumcheck.
//!
//! The ordinary closing check reads every opened value: the opening sends them, and the
//! verifier evaluates the AIRs and the bus terminal composition on them.  Its input grows
//! with the tables' widths.
//!
//! The check is a polynomial of degree at most three in the values, a sum of one
//! polynomial a table.  Under [`set_closing_sumcheck`] the pair opening proves that
//! polynomial against the committed columns instead, by a column sumcheck over each
//! table's columns, and sends no value (see the boolean trace commitment).  This module
//! supplies the polynomial: [`AirClosing`] evaluates each table's AIR constraints and bus
//! shares over [`polar::Polar`] values, which split it by degree into slots.

pub mod polar;

use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

use p3_air::symbolic::AirLayout;
use p3_air::{Air, BaseAir};
use p3_bus::{BusEvaluation, BusFactorPlan};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_sumcheck::ClosingRelation;

use self::polar::{Polar, PolarRoot};
use crate::folder::MultilinearFolder;
use crate::selectors::{BoundaryEvals, air_periodic_evals_at};
use crate::zerocheck::{BusFamily, ZerocheckError};

/// Whether proofs close their shared sumcheck by a column sumcheck; see the module docs.
static CLOSING: AtomicBool = AtomicBool::new(false);

/// Close the shared sumcheck by a column sumcheck inside the pair opening, or not.
///
/// Prover and verifier must agree; returns the previous setting.
pub fn set_closing_sumcheck(on: bool) -> bool {
    CLOSING.swap(on, Ordering::SeqCst)
}

/// Whether [`set_closing_sumcheck`] is on.
#[must_use]
pub fn closing_sumcheck() -> bool {
    CLOSING.load(Ordering::SeqCst)
}

/// An AIR the column-sumcheck closing can evaluate over split values.
pub trait ClosingAir<F: Field, EF: ExtensionField<F>>:
    Sync + for<'b> Air<MultilinearFolder<'b, F, Polar<F, EF>, PolarRoot<F, EF>>>
{
}

impl<F, EF, A> ClosingAir<F, EF> for A
where
    F: Field,
    EF: ExtensionField<F>,
    A: Sync + for<'b> Air<MultilinearFolder<'b, F, Polar<F, EF>, PolarRoot<F, EF>>>,
{
}

/// One bus share of the terminal composition: its AIR, its weight, its factor, and the
/// boundary selectors of its row point.
type Share<'a, F, EF> = (usize, EF, BusFactorPlan<'a, F, EF>, BoundaryEvals<EF>);

/// The closing check of one batch, as a [`ClosingRelation`].
///
/// ```text
///     final_sum = eq(tau, r) * sum_t beta^t C_t(v_t) + lambda * sum_shares w (factor - 1)
/// ```
///
/// Table `t`'s polynomial is its own constraints and its own bus shares, at its own point.
pub struct AirClosing<'a, F: Field, EF: ExtensionField<F>, A> {
    airs: &'a [&'a A],
    widths: Vec<(usize, usize)>,
    alpha: EF,
    lambda: EF,
    /// `eq(tau, r) * beta^t`, one a table.
    weights: Vec<EF>,
    boundary: Vec<BoundaryEvals<EF>>,
    periodic: Vec<Vec<EF>>,
    public_values: &'a [&'a [F]],
    shares: Vec<Share<'a, F, EF>>,
    target: Option<EF>,
}

/// The challenges and point of a shared sumcheck the closing check reads.
#[derive(Clone, Debug)]
pub struct ClosingChallenges<EF> {
    /// The AIR constraint batching scalar.
    pub alpha: EF,
    /// The AIR batching scalar.
    pub beta: EF,
    /// The bus family's scalar.
    pub lambda: EF,
    /// The zerocheck point.
    pub tau: Vec<EF>,
    /// The bound sumcheck point.
    pub point: Point<EF>,
}

impl<'a, F, EF, A> AirClosing<'a, F, EF, A>
where
    F: Field,
    EF: ExtensionField<F>,
    A: BaseAir<F> + ClosingAir<F, EF>,
{
    /// The closing check of `airs` at the challenges, with the bus family, if any, and the
    /// fingerprint weights its output draws, which its compiled factors borrow.
    ///
    /// # Errors
    ///
    /// A periodic declaration the AIR's height cannot hold, or a bus share that does not
    /// evaluate.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        airs: &'a [&'a A],
        log_heights: &[usize],
        challenges: &ClosingChallenges<EF>,
        public_values: &'a [&'a [F]],
        bus: Option<BusFamily<'a, F, EF>>,
        fingerprint_weights: &'a [EF],
        target: Option<EF>,
    ) -> Result<Self, ZerocheckError> {
        let max_log_height = challenges.point.num_variables();
        let eq_at_point = Point::eval_eq(&challenges.tau, challenges.point.as_slice());
        let mut widths = Vec::with_capacity(airs.len());
        let mut weights = Vec::with_capacity(airs.len());
        let mut boundary = Vec::with_capacity(airs.len());
        let mut periodic = Vec::with_capacity(airs.len());
        let mut beta_power = EF::ONE;
        for (&air, &log_height) in airs.iter().zip(log_heights) {
            let layout = AirLayout::from_air::<F>(air);
            widths.push((layout.main_width, layout.preprocessed_width));
            let point = &challenges.point.as_slice()[max_log_height - log_height..];
            boundary.push(BoundaryEvals::at(point));
            periodic.push(air_periodic_evals_at::<F, EF, _>(air, point)?);
            weights.push(eq_at_point * beta_power);
            beta_power *= challenges.beta;
        }
        let shares = match bus {
            Some(family) => family
                .context
                .closing_shares(
                    family.output,
                    fingerprint_weights,
                    challenges.lambda,
                    &challenges.point,
                )
                .map_err(ZerocheckError::BusBinding)?,
            None => Vec::new(),
        };
        let closing = Self {
            airs,
            widths,
            alpha: challenges.alpha,
            lambda: challenges.lambda,
            weights,
            boundary,
            periodic,
            public_values,
            shares,
            target,
        };
        // Every table evaluates before any proof value is read.
        for (table, &(main, preprocessed)) in closing.widths.iter().enumerate() {
            let (zeros_main, zeros_preprocessed) = (EF::zero_vec(main), EF::zero_vec(preprocessed));
            let slot = (zeros_main.as_slice(), zeros_preprocessed.as_slice());
            closing
                .try_evaluate(table, [slot, slot, slot])
                .map_err(ZerocheckError::BusBinding)?;
        }
        Ok(closing)
    }

    fn try_evaluate(
        &self,
        table: usize,
        slots: [(&[EF], &[EF]); 3],
    ) -> Result<[EF; 4], crate::bus::BusBindingError> {
        let air = self.airs[table];
        let (main_width, preprocessed_width) = self.widths[table];
        let main: Vec<Polar<F, EF>> = (0..main_width)
            .map(|j| Polar::column([slots[0].0[j], slots[1].0[j], slots[2].0[j]]))
            .collect();
        let preprocessed: Vec<Polar<F, EF>> = (0..preprocessed_width)
            .map(|j| Polar::column([slots[0].1[j], slots[1].1[j], slots[2].1[j]]))
            .collect();
        // No table reads a row ahead under the closing (the pair opening refuses one).
        let next = vec![Polar::constant(EF::ZERO); main_width];
        let preprocessed_next = vec![Polar::constant(EF::ZERO); preprocessed_width];
        let selectors = &self.boundary[table];
        let periodic: Vec<Polar<F, EF>> =
            self.periodic[table].iter().map(|&x| Polar::constant(x)).collect();
        let folder = MultilinearFolder::<F, Polar<F, EF>, PolarRoot<F, EF>>::new(
            &main,
            &next,
            BoundaryEvals {
                first: Polar::constant(selectors.first),
                last: Polar::constant(selectors.last),
                transition: Polar::constant(selectors.transition),
            },
            self.public_values[table],
            PolarRoot::constant(self.alpha),
        )
        .with_preprocessed(&preprocessed, &preprocessed_next)
        .with_periodic(&periodic);
        let mut total = folder.eval_air(air).scale(self.weights[table]);
        let mut scratch = Vec::new();
        for (owner, weight, factor, selectors) in &self.shares {
            if *owner != table {
                continue;
            }
            let values = BusEvaluation {
                main: &main,
                preprocessed: &preprocessed,
                public: self.public_values[table],
                periodic: &periodic,
                is_first_row: Polar::constant(selectors.first),
                is_last_row: Polar::constant(selectors.last),
                is_transition: Polar::constant(selectors.transition),
            };
            let term = factor.factor_minus_one_over(&mut scratch, values, Polar::constant)?;
            total += PolarRoot::from(term).scale(self.lambda * *weight);
        }
        Ok(total.parts())
    }
}

impl<F, EF, A> ClosingRelation<EF> for AirClosing<'_, F, EF, A>
where
    F: Field,
    EF: ExtensionField<F>,
    A: BaseAir<F> + ClosingAir<F, EF>,
{
    fn widths(&self) -> Vec<(usize, usize)> {
        self.widths.clone()
    }

    fn target(&self) -> Option<EF> {
        self.target
    }

    fn evaluate(&self, table: usize, slots: [(&[EF], &[EF]); 3]) -> [EF; 4] {
        self.try_evaluate(table, slots)
            .expect("every table evaluated when the closing was built")
    }
}
