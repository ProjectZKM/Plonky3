//! Bus family of the shared AIR sumcheck, binding product-GKR leaf claims to committed traces.
//!
//! For each direction, ProductGKR leaves a claim `L(q)`. This family proves
//! `L(q) - 1` equals the weighted sum of the rowwise bus factors minus one.
//! Short tables are lifted with `eq(prefix, 1^k)`, whose Boolean-cube sum is one.
//!
//! The family runs inside the zerocheck sumcheck, over the same cube and challenges.
//! Its terminal expression is checked against the same openings the AIR constraints read.
//!
//! A packed Boolean column is read where its table holds it for an AIR's first
//! [`BIT_ROUNDS`] active rounds: after `j` folds a value is a sum of `2^j` source bits, each
//! weighted by the equality polynomial of the challenges folded so far, read through tables of
//! the weights' subset sums. Only at the last of those folds is the column written out, at
//! `2^-BIT_ROUNDS` of its height; every fold after that releases the half it leaves.
//!
//! A dense column, a periodic one and the selectors are lifted into the challenge field as
//! they always were. A table shorter than the cube holds nothing while it is dormant.

use alloc::vec::Vec;

use p3_bus::{BusDirection, BusEvaluation, BusReductionOutput};
use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::generic_degree::{RoundPolyInterpolator, RoundProver};
use p3_sumcheck::layout::{ColumnView, Table};

use crate::bus::BusContext;

/// Active rounds an AIR's packed Boolean columns are read as bits before they are written out.
const BIT_ROUNDS: usize = 4;

/// Bits one subset-sum table covers.
const TABLE_BITS: usize = 8;

/// One source column of a bus term, as the rounds read it.
enum Source<'a, F: Field, EF> {
    /// A packed Boolean column, read through the challenges folded into it so far.
    Bits(ColumnView<'a, F>),
    /// The column folded into the challenge field.
    Dense(Poly<EF>),
}

/// The challenges an AIR's bit columns have absorbed, as subset sums of their equality weights.
///
/// After `j` folds, the value at `i` of a column of current length `L` reads the `2^j` source
/// bits at `b L + i`, the first challenge on the most significant bit of `b`. Table `t` holds,
/// for every byte `m`, the sum of the weights of the offsets `8 t + k` whose bit `k` of `m` is
/// set.
struct BitFolds<EF> {
    /// Challenges folded so far, first bound first.
    challenges: Vec<EF>,
    /// One subset-sum table per eight offsets.
    tables: Vec<[EF; 1 << TABLE_BITS]>,
}

impl<EF: Field> BitFolds<EF> {
    /// No challenge yet: a value is its bit.
    fn new() -> Self {
        let mut folds = Self {
            challenges: Vec::new(),
            tables: Vec::new(),
        };
        folds.rebuild();
        folds
    }

    /// Absorb one more challenge.
    fn push(&mut self, challenge: EF) {
        self.challenges.push(challenge);
        self.rebuild();
    }

    /// Rebuild the subset-sum tables for the challenges absorbed so far.
    fn rebuild(&mut self) {
        let weights = Point::new(self.challenges.clone()).equality_weights_msb();
        self.tables = weights
            .chunks(TABLE_BITS)
            .map(|chunk| {
                let mut table = [EF::ZERO; 1 << TABLE_BITS];
                for mask in 1..1usize << chunk.len() {
                    let low = mask.trailing_zeros() as usize;
                    table[mask] = table[mask & (mask - 1)] + chunk[low];
                }
                table
            })
            .collect();
    }

    /// The value at `i` of the bit column `column`, at current length `len`.
    #[inline]
    fn value<F: Field>(&self, column: ColumnView<'_, F>, len: usize, i: usize) -> EF {
        let mut value = EF::ZERO;
        for (chunk, table) in self.tables.iter().enumerate() {
            let offsets = (1usize << self.challenges.len()).min((chunk + 1) * TABLE_BITS);
            let mut mask = 0usize;
            for (k, offset) in (chunk * TABLE_BITS..offsets).enumerate() {
                mask |= usize::from(column.value(offset * len + i) == F::ONE) << k;
            }
            value += table[mask];
        }
        value
    }
}

impl<F: Field, EF: ExtensionField<F>> Source<'_, F, EF> {
    /// The column at `i`, of current length `len`.
    #[inline]
    fn value(&self, folds: &BitFolds<EF>, len: usize, i: usize) -> EF {
        match self {
            Self::Bits(column) => folds.value(*column, len, i),
            Self::Dense(poly) => poly.as_slice()[i],
        }
    }

    /// A column read in place when it holds bits, lifted into the challenge field otherwise.
    fn of(column: ColumnView<'_, F>) -> Source<'_, F, EF> {
        match column {
            ColumnView::Boolean { .. } => Source::Bits(column),
            ColumnView::Dense(_) => {
                Source::Dense(Poly::new(column.values().map(Into::into).collect()))
            }
        }
    }
}

/// Fold a dense polynomial's leading variable and give back the half it leaves.
fn fold_released<EF: Field>(poly: &mut Poly<EF>, challenge: EF) {
    poly.fix_prefix_var_mut(challenge);
    let mut values = core::mem::replace(poly, Poly::new(alloc::vec![EF::ZERO])).into_evals();
    values.shrink_to_fit();
    *poly = Poly::new(values);
}

/// Prover state for the mixed-height bus composition polynomial.
pub(crate) struct BusCompositionProver<'a, F: Field, EF: ExtensionField<F>> {
    /// Checked declarations and physical layout used by every evaluation.
    context: &'a BusContext<F, EF>,
    /// Tuple equality coefficients sampled after commitment.
    fingerprint_weights: Vec<EF>,
    /// Random tuple-fingerprint shift.
    offset: EF,
    /// Folded source polynomials grouped once per AIR.
    airs: Vec<AirState<'a, F, EF>>,
    /// Maximum round degree, derived once from the public plan.
    degree: usize,
    /// Number of global variables already bound.
    round: usize,
}

/// One bus term evaluated from an AIR's shared folded columns.
struct CompositionTerm<EF> {
    /// Coordinates of the symbolic declaration in the bus context.
    owner: p3_bus::BusBlockOwner,
    /// Named bus selecting the tuple-domain slots.
    bus: usize,
    /// Fixed coefficient from direction batching and ProductGKR block selection.
    coefficient: EF,
    /// Cube sum of the unweighted row composition before this AIR activates.
    row_claim: EF,
}

/// Folded source multilinears shared by every bus term owned by one AIR.
struct AirState<'a, F: Field, EF: ExtensionField<F>> {
    /// The main columns this AIR's declarations read.
    main: Vec<Source<'a, F, EF>>,
    /// Column index of each of those, and the declared width they are placed into.
    main_layout: (Vec<usize>, usize),
    /// The preprocessed columns this AIR's declarations read.
    preprocessed: Vec<Source<'a, F, EF>>,
    /// Column index of each of those, and the declared width they are placed into.
    preprocessed_layout: (Vec<usize>, usize),
    /// The periodic columns this AIR's declarations read, at full height.
    periodic: Vec<Source<'a, F, EF>>,
    /// Column index of each of those, and the declared width they are placed into.
    periodic_layout: (Vec<usize>, usize),
    /// First-row, last-row, and transition selector polynomials.
    selectors: [Poly<EF>; 3],
    /// Equality polynomial anchored at the ProductGKR row point.
    equality: Poly<EF>,
    /// Bus terms emitted by this AIR.
    terms: Vec<CompositionTerm<EF>>,
    /// Number of global prefix variables absent from this AIR table.
    unused_prefix: usize,
    /// Evaluation of the fixed all-one-vertex selector on bound prefix coordinates.
    prefix_evaluation: EF,
    /// Public inputs read by this block's symbolic expressions.
    public_values: Vec<F>,
    /// The challenges the bit columns have absorbed, while any remains bits.
    folds: BitFolds<EF>,
    /// Active folds this AIR's columns have taken.
    active_folds: usize,
}

impl<F, EF> AirState<'_, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Compute every term's initial cube sum in one pass over the shared AIR columns.
    ///
    /// The result is read only while the global prefix is still ahead of this table.
    fn initialize_claims(
        &mut self,
        context: &BusContext<F, EF>,
        fingerprint_weights: &[EF],
        offset: EF,
    ) {
        let height = self.equality.as_slice().len();

        // Slot placement is settled once per term, outside the row loop below.
        let factors = self
            .terms
            .iter()
            .map(|term| {
                context
                    .plan()
                    .compile_factor(
                        term.bus,
                        context.interaction(term.owner),
                        fingerprint_weights,
                        offset,
                    )
                    .expect("a checked bus plan compiles against its own declarations")
            })
            .collect::<Vec<_>>();

        let folds = &self.folds;
        let main_polys = &self.main;
        let (main_indices, main_width) = (&self.main_layout.0, self.main_layout.1);
        let fixed_polys = &self.preprocessed;
        let (fixed_indices, fixed_width) =
            (&self.preprocessed_layout.0, self.preprocessed_layout.1);
        let periodic_polys = &self.periodic;
        let (periodic_indices, periodic_width) = (&self.periodic_layout.0, self.periodic_layout.1);
        let selectors = &self.selectors;
        let equality = &self.equality;
        let public_values = &self.public_values;
        let claims = (0..height)
            .into_par_iter()
            .par_fold_reduce(
                || {
                    (
                        EF::zero_vec(factors.len()),
                        EF::zero_vec(main_width),
                        EF::zero_vec(fixed_width),
                        EF::zero_vec(periodic_width),
                        Vec::new(),
                    )
                },
                |(mut claims, mut main, mut preprocessed, mut periodic, mut scratch), row| {
                    // Unread columns keep their zero, which no planned expression names.
                    for (&index, column) in main_indices.iter().zip(main_polys) {
                        main[index] = column.value(folds, height, row);
                    }
                    for (&index, column) in fixed_indices.iter().zip(fixed_polys) {
                        preprocessed[index] = column.value(folds, height, row);
                    }
                    for (&index, column) in periodic_indices.iter().zip(periodic_polys) {
                        periodic[index] = column.value(folds, height, row);
                    }
                    let evaluation = BusEvaluation {
                        main: &main,
                        preprocessed: &preprocessed,
                        public: public_values,
                        periodic: &periodic,
                        is_first_row: selectors[0].as_slice()[row],
                        is_last_row: selectors[1].as_slice()[row],
                        is_transition: selectors[2].as_slice()[row],
                    };
                    let weight = equality.as_slice()[row];
                    for (claim, factor) in claims.iter_mut().zip(&factors) {
                        let value = factor
                            .evaluate::<EF>(&mut scratch, evaluation)
                            .expect("a planned expression resolves against its owning table");
                        *claim += weight * (value - EF::ONE);
                    }
                    (claims, main, preprocessed, periodic, scratch)
                },
                |(mut left, main, preprocessed, periodic, scratch), (right, ..)| {
                    for (claim, partial) in left.iter_mut().zip(right) {
                        *claim += partial;
                    }
                    (left, main, preprocessed, periodic, scratch)
                },
            )
            .0;

        for (term, claim) in self.terms.iter_mut().zip(claims) {
            term.row_claim = claim;
        }
    }
}

impl<F, EF> AirState<'_, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Bind this AIR's leading row variable.
    ///
    /// Dense columns fold and give back the half they leave. Bit columns absorb the challenge,
    /// and at the last of their bit rounds are written out at the length the folds left.
    fn fold(&mut self, challenge: EF) {
        let len = self.equality.as_slice().len() / 2;
        for poly in self
            .selectors
            .iter_mut()
            .chain(core::iter::once(&mut self.equality))
        {
            fold_released(poly, challenge);
        }
        self.folds.push(challenge);
        self.active_folds += 1;
        let write_out = self.active_folds == BIT_ROUNDS;
        let folds = &self.folds;
        for column in self
            .main
            .iter_mut()
            .chain(&mut self.preprocessed)
            .chain(&mut self.periodic)
        {
            match column {
                Source::Dense(poly) => fold_released(poly, challenge),
                Source::Bits(view) if write_out => {
                    let view = *view;
                    *column = Source::Dense(Poly::new(
                        (0..len)
                            .into_par_iter()
                            .map(|i| folds.value(view, len, i))
                            .collect(),
                    ));
                }
                Source::Bits(_) => {}
            }
        }
    }
}

impl<'a, F, EF> BusCompositionProver<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Build the formal polynomial whose cube sum must equal the ProductGKR claims.
    ///
    /// # Arguments
    ///
    /// - `num_variables`: width of the shared cube, at least the tallest bus table.
    /// - `periodic`: the tables [`BusContext::periodic_tables`] returns.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        context: &'a BusContext<F, EF>,
        output: &BusReductionOutput<EF>,
        tables: &[&'a Table<F>],
        preprocessed: &[Option<&'a Table<F>>],
        periodic: &[Option<Table<F>>],
        public_values: &[&[F]],
        direction_challenge: EF,
        num_variables: usize,
    ) -> Self {
        debug_assert!(num_variables >= context.max_num_variables());
        let weights = output.challenges.fingerprint_weights();
        let mut airs = (0..tables.len()).map(|_| None).collect::<Vec<_>>();

        for direction in BusDirection::ALL {
            let direction_weight = match direction {
                BusDirection::Push => EF::ONE,
                BusDirection::Pull => direction_challenge,
            };
            for share in context.plan().terminal_shares(direction) {
                let air = share.owner.air;
                let row_point = &output.product.point[share.prefix_variables..];
                let block_weight = share
                    .prefix_weight(&output.product.point)
                    .expect("a planned share addresses its own product-tree point");
                let coefficient = direction_weight * block_weight;
                let state = airs[air].get_or_insert_with(|| {
                    // Column views read a packed Boolean table without expanding it first.
                    // Only the columns a declaration reads are taken, and a bit column is read
                    // where it lies until its first rounds are folded.
                    let main_columns = context.main_columns(air);
                    let main = main_columns
                        .iter()
                        .map(|&column| Source::of(tables[air].column(column)))
                        .collect::<Vec<_>>();
                    let fixed_columns = context.preprocessed_columns(air);
                    let fixed_width = preprocessed[air].map_or(0, Table::num_polys);
                    let preprocessed = preprocessed[air]
                        .iter()
                        .flat_map(|&table| {
                            fixed_columns
                                .iter()
                                .map(move |&column| Source::of(table.column(column)))
                        })
                        .collect::<Vec<_>>();
                    // Periodic columns fold exactly like committed ones.
                    let periodic_columns = context.periodic_columns(air);
                    let periodic_width = periodic[air].as_ref().map_or(0, Table::num_polys);
                    let periodic = periodic[air]
                        .iter()
                        .flat_map(|table| {
                            periodic_columns.iter().map(|&column| {
                                Source::Dense(Poly::new(
                                    table.column(column).values().map(Into::into).collect(),
                                ))
                            })
                        })
                        .collect::<Vec<_>>();
                    let height = 1usize << share.row_variables;
                    let selectors = [
                        Poly::new((0..height).map(|row| EF::from_bool(row == 0)).collect()),
                        Poly::new(
                            (0..height)
                                .map(|row| EF::from_bool(row + 1 == height))
                                .collect(),
                        ),
                        Poly::new(
                            (0..height)
                                .map(|row| EF::from_bool(row + 1 < height))
                                .collect(),
                        ),
                    ];
                    AirState {
                        main,
                        main_layout: (main_columns.to_vec(), tables[air].num_polys()),
                        preprocessed,
                        preprocessed_layout: (fixed_columns.to_vec(), fixed_width),
                        periodic,
                        periodic_layout: (periodic_columns.to_vec(), periodic_width),
                        selectors,
                        equality: Poly::new(Point::new(row_point).equality_weights_msb()),
                        terms: Vec::new(),
                        unused_prefix: num_variables - share.row_variables,
                        prefix_evaluation: EF::ONE,
                        public_values: public_values[air].to_vec(),
                        folds: BitFolds::new(),
                        active_folds: 0,
                    }
                });
                // Row geometry is captured from the first share and reused by every later one.
                debug_assert_eq!(
                    state.unused_prefix,
                    num_variables - share.row_variables,
                    "every block of one AIR shares its trace height"
                );
                state.terms.push(CompositionTerm {
                    owner: share.owner,
                    bus: share.bus,
                    coefficient,
                    row_claim: EF::ZERO,
                });
            }
        }

        // A table as tall as the statement never reads its own claim, so it never pays for one.
        for air in airs
            .iter_mut()
            .flatten()
            .filter(|air| air.unused_prefix > 0)
        {
            air.initialize_claims(context, &weights, output.challenges.offset);
        }

        Self {
            context,
            fingerprint_weights: weights,
            offset: output.challenges.offset,
            airs: airs.into_iter().flatten().collect(),
            degree: context.composition_degree(),
            round: 0,
        }
    }

    fn evaluate_air(&self, air: &AirState<'_, F, EF>, node: EF) -> EF {
        // Slot placement is settled once per term, outside the row loop below.
        let factors = air
            .terms
            .iter()
            .map(|term| {
                self.context
                    .plan()
                    .compile_factor(
                        term.bus,
                        self.context.interaction(term.owner),
                        &self.fingerprint_weights,
                        self.offset,
                    )
                    .expect("a checked bus plan compiles against its own declarations")
            })
            .collect::<Vec<_>>();

        // Interpolate shared columns once, then evaluate every declaration owned by this AIR.
        let len = air.equality.as_slice().len();
        let half = len / 2;
        (0..half)
            .into_par_iter()
            .map_init(
                || {
                    (
                        EF::zero_vec(air.main_layout.1),
                        EF::zero_vec(air.preprocessed_layout.1),
                        EF::zero_vec(air.periodic_layout.1),
                        Vec::new(),
                    )
                },
                |(main, prep, periodic, scratch), row| {
                    let interpolate = |poly: &Poly<EF>| {
                        let values = poly.as_slice();
                        values[row] + (values[row + half] - values[row]) * node
                    };
                    let source = |column: &Source<'_, F, EF>| {
                        let low = column.value(&air.folds, len, row);
                        low + (column.value(&air.folds, len, row + half) - low) * node
                    };
                    // Unread columns keep their zero, which no planned expression names.
                    for (&index, column) in air.main_layout.0.iter().zip(&air.main) {
                        main[index] = source(column);
                    }
                    for (&index, column) in air.preprocessed_layout.0.iter().zip(&air.preprocessed)
                    {
                        prep[index] = source(column);
                    }
                    for (&index, column) in air.periodic_layout.0.iter().zip(&air.periodic) {
                        periodic[index] = source(column);
                    }
                    let evaluation = BusEvaluation {
                        main,
                        preprocessed: prep,
                        public: &air.public_values,
                        periodic,
                        is_first_row: interpolate(&air.selectors[0]),
                        is_last_row: interpolate(&air.selectors[1]),
                        is_transition: interpolate(&air.selectors[2]),
                    };
                    let equality = interpolate(&air.equality);
                    air.terms
                        .iter()
                        .zip(&factors)
                        .map(|(term, factor)| {
                            let value = factor
                                .evaluate::<EF>(scratch, evaluation)
                                .expect("a planned expression resolves against its folded table");
                            term.coefficient * equality * (value - EF::ONE)
                        })
                        .sum::<EF>()
                },
            )
            .sum()
    }
}

impl<F, EF> RoundProver<EF> for BusCompositionProver<'_, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn fold(&mut self, challenge: EF) {
        // Dormant blocks evaluate one more coordinate of χ_k at the all-one vertex.
        for air in &mut self.airs {
            if self.round < air.unused_prefix {
                air.prefix_evaluation *= challenge;
                continue;
            }
            air.fold(challenge);
        }
        self.round += 1;
    }

    fn round_poly(&self) -> Vec<EF> {
        // Generic-degree encoding omits node one, which the verifier derives from the claim.
        (0..self.degree)
            .map(RoundPolyInterpolator::<EF>::transmitted_node)
            .map(|node| {
                self.airs
                    .iter()
                    .map(|air| {
                        let body = if self.round < air.unused_prefix {
                            // χ_k contributes the current node; its remaining cube sum is one.
                            node * air
                                .terms
                                .iter()
                                .map(|term| term.coefficient * term.row_claim)
                                .sum::<EF>()
                        } else {
                            // Active terms share one interpolation of their AIR columns.
                            self.evaluate_air(air, node)
                        };
                        air.prefix_evaluation * body
                    })
                    .sum()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use p3_air::symbolic::AirLayout;
    use p3_air::{Air, BaseAir, WindowAccess};
    use p3_baby_bear::BabyBear;
    use p3_binary_field::BinaryField128;
    use p3_bus::{BusActivation, BusDirection, BusInteractionBuilder, BusName, BusSymbolicBuilder};
    use p3_field::PrimeCharacteristicRing;
    use p3_sumcheck::generic_degree::RoundPolyInterpolator;

    struct TransitionBusAir;

    impl BaseAir<BabyBear> for TransitionBusAir {
        fn width(&self) -> usize {
            // One column supplies the payload used by the transition-weighted tuple.
            1
        }
    }

    impl<AB: BusInteractionBuilder<F = BabyBear>> Air<AB> for TransitionBusAir {
        fn eval(&self, builder: &mut AB) {
            // Multiplying by the transition selector exposes its round-degree contribution.
            let value: AB::Expr = builder.main().current_slice()[0].into();
            builder.push_bus_interaction(
                BusName::new("transition"),
                BusDirection::Push,
                [builder.is_transition() * value],
                BusActivation::Always,
            );
        }
    }

    #[test]
    fn transition_selectors_contribute_one_to_each_round_degree() {
        let profile: BusSymbolicBuilder<BabyBear> =
            BusSymbolicBuilder::from_air(&TransitionBusAir, AirLayout::from_air(&TransitionBusAir));

        assert_eq!(
            profile.interactions()[0].factor_degree_multiple_with_transition(1),
            2
        );
    }

    #[test]
    fn bit_columns_read_what_their_dense_folds_hold() {
        use p3_matrix::dense::RowMajorMatrix;
        use p3_multilinear_util::poly::Poly;
        use p3_sumcheck::layout::Table;
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        use super::{BitFolds, Source};

        type F = BinaryField128;
        let mut rng = SmallRng::seed_from_u64(7);
        let height = 1 << 7;
        let cells: alloc::vec::Vec<F> = (0..2 * height)
            .map(|_| F::from_bool(rng.random::<bool>()))
            .collect();
        let table = Table::from_boolean_rows(&RowMajorMatrix::new(cells, 2)).unwrap();
        for column in 0..2 {
            let view = table.column(column);
            let bits: Source<'_, F, F> = Source::of(view);
            assert!(matches!(bits, Source::Bits(_)));
            let mut dense = Poly::new(view.values().collect::<alloc::vec::Vec<_>>());
            let mut folds = BitFolds::new();
            let mut len = height;
            // Past the write-out round too, so a column read as bits that long still agrees.
            for _ in 0..6 {
                for i in 0..len {
                    assert_eq!(bits.value(&folds, len, i), dense.as_slice()[i]);
                }
                let challenge = F::from_u64(rng.random::<u64>());
                dense.fix_prefix_var_mut(challenge);
                folds.push(challenge);
                len /= 2;
            }
        }
    }

    #[test]
    fn round_nodes_are_distinct_in_characteristic_two() {
        // Integer embedding would map node two back to zero in characteristic two.
        let nodes = (0..5)
            .map(RoundPolyInterpolator::<BinaryField128>::transmitted_node)
            .collect::<alloc::vec::Vec<_>>();

        assert_eq!(nodes[0], BinaryField128::ZERO);
        assert_ne!(nodes[1], BinaryField128::ZERO);
        for (index, node) in nodes.iter().enumerate() {
            assert!(!nodes[index + 1..].contains(node));
        }
    }
}
