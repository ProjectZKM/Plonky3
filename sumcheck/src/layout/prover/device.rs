//! The suffix layout's opening of a lone whole-space column, with the column held on a device.
//!
//! The column is the stacked polynomial itself, so every claim is an evaluation of it at a
//! point over every stacked variable, and the batched weight is a sum of equality tables.
//! The device evaluates the polynomial, builds the weights, and plays the sumcheck rounds over
//! the pair; this driver plays the transcript [`SuffixProver`](super::SuffixProver) plays for
//! the same claims, so the proof is the same bytes.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use crate::layout::LayoutStrategy;
use crate::layout::transcript::{
    BatchingShape, LayoutBinding, OpeningProverTranscript, OpeningShape, PointSource,
    VirtualProverTranscript, VirtualShape, prover_batching_challenge,
};
use crate::product_polynomial::ProductPolynomial;
use crate::strategy::{Basis, SumcheckProver, VariableOrder};
use crate::table::{OpeningEvals, OpeningRequest, TableShape};
use crate::transcript::{ProverTranscript, SumcheckShape};
use crate::{SumcheckData, extrapolate_01inf};

/// A multilinear polynomial over every stacked variable, held on a device, and the weights it
/// is sumchecked against.
///
/// Points list their coordinates most significant variable first, as [`Poly`] reads them.
/// Every round binds the last unbound variable, the least significant index bit.
pub trait StackedDevice<F: Field, EF: ExtensionField<F>> {
    /// The polynomial at `point`.
    fn eval(&mut self, point: &Point<EF>) -> EF;

    /// Sets the weights to `sum_i c_i eq(z_i, x)` over the `(z_i, c_i)` of `claims`.
    fn weigh(&mut self, claims: &[(Point<EF>, EF)]);

    /// The round polynomial of `sum_x f(x) w(x)` in the last variable: its value at zero and
    /// its leading coefficient.
    fn round(&mut self) -> (EF, EF);

    /// Binds the last variable of the polynomial and of the weights to `r`.
    fn fold(&mut self, r: EF);

    /// The polynomial and the weights as they now stand.
    fn residual(&mut self) -> (Poly<EF>, Poly<EF>);
}

/// The suffix layout over one whole-space column held by a device, which every step that
/// reads the column is handed.
pub struct DeviceSuffix<F: Field, EF: ExtensionField<F>> {
    num_variables: usize,
    folding: usize,
    /// Every concrete claim's point and value, in recording order.
    claims: Vec<(Point<EF>, EF)>,
    /// Every out-of-domain claim's point and value, in recording order.
    virtual_claims: Vec<(Point<EF>, EF)>,
    _base: PhantomData<F>,
}

impl<F: Field, EF: ExtensionField<F>> DeviceSuffix<F, EF> {
    /// The layout of one column of `num_variables` variables, its first `folding` rounds
    /// played before the residual.
    pub fn new(num_variables: usize, folding: usize) -> Self {
        assert!(folding <= num_variables);
        Self {
            num_variables,
            folding,
            claims: Vec::new(),
            virtual_claims: Vec::new(),
            _base: PhantomData,
        }
    }

    /// The one table's shape.
    pub fn table_shapes(&self) -> Vec<TableShape> {
        alloc::vec![TableShape::new(self.num_variables, 1)]
    }

    fn binding(&self) -> LayoutBinding {
        LayoutBinding::new(
            self.num_variables,
            LayoutStrategy::new(false, VariableOrder::Suffix),
            self.table_shapes(),
        )
    }

    /// Concrete claims recorded so far.
    pub fn num_claims(&self) -> usize {
        self.claims.len()
    }

    /// Rounds played before the residual.
    pub const fn folding(&self) -> usize {
        self.folding
    }

    /// As [`Layout::add_virtual_eval`](super::Layout::add_virtual_eval).
    pub fn add_virtual_eval<Ch>(
        &mut self,
        device: &mut dyn StackedDevice<F, EF>,
        challenger: &mut Ch,
    ) -> EF
    where
        F: TranscriptField,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let shape = VirtualShape::new(self.binding());
        let mut transcript = VirtualProverTranscript::<Ch, F, EF>::new(challenger, shape);
        let point = transcript.point();
        let eval = device.eval(&point);
        transcript.evaluation(eval);
        transcript.finish();
        self.virtual_claims.push((point, eval));
        eval
    }

    /// As [`Layout::eval_at_known`](super::Layout::eval_at_known), for the column's current
    /// row alone.
    ///
    /// # Panics
    ///
    /// Unless the request names the one column's current row, and its value.
    pub fn eval_at_known<Ch>(
        &mut self,
        table_idx: usize,
        batch: &OpeningRequest,
        point: &Point<EF>,
        evals: &OpeningEvals<EF>,
        challenger: &mut Ch,
    ) where
        F: TranscriptField,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        assert_eq!(table_idx, 0, "the layout holds one table");
        assert!(
            batch.current() == [0] && batch.next().is_empty(),
            "a device opening reads the one column's current row"
        );
        assert!(batch.has_same_shape(evals));
        assert_eq!(point.num_variables(), self.num_variables);
        let shape = OpeningShape::new(
            self.binding(),
            table_idx,
            self.num_variables,
            batch.current(),
            batch.next(),
            PointSource::Given,
        );
        let mut transcript = OpeningProverTranscript::<Ch, F, EF>::new(challenger, shape);
        transcript.evaluations(evals);
        transcript.finish();
        self.claims.push((point.clone(), evals.current()[0]));
    }

    /// As [`Layout::into_sumcheck`](super::Layout::into_sumcheck), with the residual left on
    /// the device: the batching draw and the preprocessing rounds, returning the claimed sum
    /// of the residual and the challenges drawn.
    pub fn into_rounds<Ch>(
        self,
        device: &mut dyn StackedDevice<F, EF>,
        sumcheck_data: &mut SumcheckData<F, EF>,
        pow_bits: usize,
        challenger: &mut Ch,
    ) -> (EF, Point<EF>)
    where
        F: TranscriptField,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let shape = BatchingShape::new(
            self.binding(),
            self.claims.len(),
            self.virtual_claims.len(),
        );
        let alpha: EF = prover_batching_challenge::<Ch, F, EF>(challenger, &shape);

        // Concrete claims take the low powers in recording order, out-of-domain ones the rest.
        let weighted: Vec<(Point<EF>, EF)> = self
            .claims
            .iter()
            .chain(&self.virtual_claims)
            .zip(alpha.powers())
            .map(|((point, _), power)| (point.clone(), power))
            .collect();
        let mut sum: EF = self
            .claims
            .iter()
            .chain(&self.virtual_claims)
            .zip(alpha.powers())
            .map(|((_, eval), power)| *eval * power)
            .sum();
        device.weigh(&weighted);

        let point = fold_rounds(
            device,
            &mut sum,
            sumcheck_data,
            challenger,
            self.folding,
            pow_bits,
        );
        (sum, point)
    }
}

/// Plays `rounds` sumcheck rounds of `sum = sum_x f(x) w(x)` over the pair `device` holds,
/// each followed by `pow_bits` of grinding, binding the last variable each round and carrying
/// `sum` to the bound pair's.
pub fn fold_rounds<F, EF, Ch>(
    device: &mut dyn StackedDevice<F, EF>,
    sum: &mut EF,
    sumcheck_data: &mut SumcheckData<F, EF>,
    challenger: &mut Ch,
    rounds: usize,
    pow_bits: usize,
) -> Point<EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let mut rs = Vec::with_capacity(rounds);
    let shape = SumcheckShape::new(rounds, pow_bits, Basis::Evaluation);
    let mut transcript = ProverTranscript::<Ch, F, EF>::new(challenger, shape);
    for _ in 0..rounds {
        let (c0, c_inf) = device.round();
        let r = sumcheck_data.observe_and_sample(&mut transcript, c0, c_inf);
        *sum = extrapolate_01inf(c0, *sum - c0, c_inf, r);
        device.fold(r);
        rs.push(r);
    }
    transcript.finish();
    Point::new(rs)
}

/// The host prover of the residual `device` holds, its claimed sum `sum`.
pub fn residual_prover<F: Field, EF: ExtensionField<F>>(
    device: &mut dyn StackedDevice<F, EF>,
    sum: EF,
) -> SumcheckProver<F, EF> {
    let (evals, weights) = device.residual();
    let poly = ProductPolynomial::new_unpacked(VariableOrder::Suffix, evals, weights);
    SumcheckProver::new(poly, sum)
}
