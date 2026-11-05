//! Typed transcript for batched Boolean trace column openings.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, CanSampleUniformBits, GrindingChallenger};
use p3_field::{AlgebraIdentity, ExtensionField, Field};
use p3_multilinear_util::point::Point;

const VERSION: u8 = 1;
const NAME: &[u8] = b"p3-boolean-trace-column-batching";
const POINT: &str = "row_point";
const VALUES: &str = "column_values";
const NEXT_VALUES: &str = "next_column_values";
const COLUMN_POINT: &str = "column_point";
const PACKING_WEIGHT: &str = "packing_weight";
const COLUMN_ROUND: &str = "column_round";
const COLUMN_CHALLENGE: &str = "column_challenge";
const FOLDED_COLUMNS: &str = "folded_columns";
const PACKED_LABEL: &[u8] = b"packed";
const SUCCESSOR_LABEL: &[u8] = b"successor";

type Alphabet<F> = FieldUnit<F>;

/// Public shape of one optimized column-batched opening run.
///
/// Every batch claims the table's whole width at its own row point, and, when `next`
/// holds, the same width one row ahead. Both value runs are bound before the batch
/// draws the column point that combines them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ColumnBatchShape {
    pub(crate) table_variables: usize,
    pub(crate) width: usize,
    pub(crate) num_batches: usize,
    pub(crate) next: bool,
    /// whir-gc experiment: values a batch sends when its columns are read through linear
    /// forms, zero when every column is sent.  Such a batch binds its row point and its
    /// forms; one sumcheck shared by every such run (see [`FormsSumcheckShape`]) then
    /// reduces all their forms to one fold of each table's columns.
    pub(crate) forms: usize,
}

impl ColumnBatchShape {
    pub(crate) const fn column_variables(self) -> usize {
        self.width.next_power_of_two().trailing_zeros() as usize
    }

    /// Views each batch reads: the current row, and the next one when it is asked for.
    pub(crate) fn num_views(self) -> usize {
        1 + usize::from(self.next)
    }

    /// Values each batch carries: the whole width once per view, or its forms.
    pub(crate) fn values_per_batch(self) -> usize {
        if self.forms > 0 { self.forms } else { self.width * self.num_views() }
    }

    fn pattern<F, EF>(self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: Field + AlgebraIdentity<F>,
    {
        let per_batch = 3 + usize::from(self.next);
        let mut steps = Vec::with_capacity(self.num_batches * per_batch);
        if self.forms > 0 {
            for _ in 0..self.num_batches {
                steps.push(Interaction::algebra::<F, EF>(
                    Hierarchy::Atomic,
                    Kind::Message,
                    POINT,
                    Length::Fixed(self.table_variables),
                ));
                steps.push(Interaction::algebra::<F, EF>(
                    Hierarchy::Atomic,
                    Kind::Message,
                    VALUES,
                    Length::Fixed(self.forms),
                ));
            }
            return InteractionPattern::new(steps).expect("flat forms transcript is valid");
        }
        for _ in 0..self.num_batches {
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                POINT,
                Length::Fixed(self.table_variables),
            ));
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                VALUES,
                Length::Fixed(self.width),
            ));
            if self.next {
                steps.push(Interaction::algebra::<F, EF>(
                    Hierarchy::Atomic,
                    Kind::Message,
                    NEXT_VALUES,
                    Length::Fixed(self.width),
                ));
            }
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                COLUMN_POINT,
                Length::Fixed(self.column_variables()),
            ));
        }
        InteractionPattern::new(steps).expect("flat column-batching transcript is valid")
    }

    fn separator<F, EF>(self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: Field + AlgebraIdentity<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());
        separator
            .instance(&(self.table_variables as u64).to_be_bytes())
            .instance(&(self.width as u64).to_be_bytes())
            .instance(&(self.num_batches as u64).to_be_bytes());
        // A run reading one row ahead carries its own label, so the two seeds part.
        if self.next {
            separator.instance(SUCCESSOR_LABEL);
        }
        if self.forms > 0 {
            separator.instance(PACKED_LABEL).instance(&(self.forms as u64).to_be_bytes());
        }
        separator
    }
}

/// Prover-side driver for one optimized opening run.
///
/// `F` is the sponge's alphabet and `EF` the field the points and values live in.
pub(crate) struct ColumnBatchProverTranscript<'a, F: TranscriptField, EF, C> {
    state: ProverState<&'a mut C, Alphabet<F>>,
    shape: ColumnBatchShape,
    _challenge: PhantomData<EF>,
}

impl<'a, F, EF, C> ColumnBatchProverTranscript<'a, F, EF, C>
where
    F: Field + TranscriptField + AlgebraIdentity<F>,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F> + GrindingChallenger<Witness = F>,
{
    pub(crate) fn new(challenger: &'a mut C, shape: ColumnBatchShape) -> Self {
        let separator = shape.separator::<F, EF>();
        Self {
            state: ProverState::new(challenger, &separator),
            shape,
            _challenge: PhantomData,
        }
    }

    pub(crate) fn batch(&mut self, point: &Point<EF>, values: &[EF], next: &[EF]) -> Point<EF> {
        debug_assert_eq!(point.num_variables(), self.shape.table_variables);
        debug_assert_eq!(values.len(), self.shape.width);
        debug_assert_eq!(next.len(), self.shape.width * usize::from(self.shape.next));
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(POINT, point.as_slice());
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(VALUES, values);
        if self.shape.next {
            self.state
                .observe_extensions::<F, EF, FieldToFieldCodec<F>>(NEXT_VALUES, next);
        }
        let coordinates = self
            .state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(
                COLUMN_POINT,
                self.shape.column_variables(),
            )
            .into_iter()
            .map(|coordinate| coordinate.into_inner());
        Point::new(coordinates.collect())
    }

    /// Bind a forms batch's row point and forms.
    pub(crate) fn bind_forms(&mut self, point: &Point<EF>, forms: &[EF]) {
        debug_assert_eq!(forms.len(), self.shape.forms);
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(POINT, point.as_slice());
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(VALUES, forms);
    }

    pub(crate) fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "column-batching transcript carries no wire values",
        );
    }
}

/// Verifier-side driver for one optimized opening run.
///
/// `F` is the sponge's alphabet and `EF` the field the points and values live in.
pub(crate) struct ColumnBatchVerifierTranscript<'a, F: TranscriptField, EF, C> {
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    shape: ColumnBatchShape,
    _challenge: PhantomData<EF>,
}

impl<'a, F, EF, C> ColumnBatchVerifierTranscript<'a, F, EF, C>
where
    F: Field + TranscriptField + AlgebraIdentity<F>,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F> + GrindingChallenger<Witness = F>,
{
    pub(crate) fn new(challenger: &'a mut C, shape: ColumnBatchShape) -> Self {
        let separator = shape.separator::<F, EF>();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            _challenge: PhantomData,
        }
    }

    pub(crate) fn batch(
        &mut self,
        point: &Point<EF>,
        values: &[EF],
        next: &[EF],
    ) -> Result<Point<EF>, p3_challenger::fs::TranscriptError> {
        debug_assert_eq!(point.num_variables(), self.shape.table_variables);
        debug_assert_eq!(values.len(), self.shape.width);
        debug_assert_eq!(next.len(), self.shape.width * usize::from(self.shape.next));
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(POINT, point.as_slice())?;
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(VALUES, values)?;
        if self.shape.next {
            self.state
                .observe_extensions::<F, EF, FieldToFieldCodec<F>>(NEXT_VALUES, next)?;
        }
        let coordinates = self
            .state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(
                COLUMN_POINT,
                self.shape.column_variables(),
            )
            .into_iter()
            .map(|coordinate| coordinate.into_inner());
        Ok(Point::new(coordinates.collect()))
    }

    /// Bind a forms batch's row point and forms.
    pub(crate) fn bind_forms(
        &mut self,
        point: &Point<EF>,
        forms: &[EF],
    ) -> Result<(), p3_challenger::fs::TranscriptError> {
        debug_assert_eq!(forms.len(), self.shape.forms);
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(POINT, point.as_slice())?;
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(VALUES, forms)?;
        Ok(())
    }

    pub(crate) fn finish(self) {
        self.state
            .finalize()
            .expect("column-batching transcript is complete");
    }
}

const FORMS_NAME: &[u8] = b"p3-boolean-trace-forms-sumcheck";

/// whir-gc experiment: the one sumcheck every forms batch of an opening shares.
///
/// Every forms batch has bound its row point and forms in its own run's transcript.
/// This one draws the weight `rho` over all their forms in run order, runs one degree-two
/// sumcheck over the widest table's column variables (each table's slots zero above its
/// own width, so it reads the trailing coordinates), and binds each table's fold.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FormsSumcheckShape {
    pub(crate) column_variables: usize,
    pub(crate) tables: usize,
}

impl FormsSumcheckShape {
    fn pattern<F, EF>(self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: Field + AlgebraIdentity<F>,
    {
        let mut steps = Vec::with_capacity(2 + 2 * self.column_variables);
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            PACKING_WEIGHT,
            Length::Scalar,
        ));
        for _ in 0..self.column_variables {
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                COLUMN_ROUND,
                Length::Fixed(2),
            ));
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                COLUMN_CHALLENGE,
                Length::Scalar,
            ));
        }
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            FOLDED_COLUMNS,
            Length::Fixed(self.tables),
        ));
        InteractionPattern::new(steps).expect("flat forms sumcheck transcript is valid")
    }

    fn separator<F, EF>(self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: Field + AlgebraIdentity<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, FORMS_NAME, self.pattern::<F, EF>());
        separator
            .instance(&(self.column_variables as u64).to_be_bytes())
            .instance(&(self.tables as u64).to_be_bytes())
            .instance(PACKED_LABEL);
        separator
    }
}

/// Prover side of [`FormsSumcheckShape`].
pub(crate) struct FormsSumcheckProverTranscript<'a, F: TranscriptField, EF, C> {
    state: ProverState<&'a mut C, Alphabet<F>>,
    _challenge: PhantomData<EF>,
}

impl<'a, F, EF, C> FormsSumcheckProverTranscript<'a, F, EF, C>
where
    F: Field + TranscriptField + AlgebraIdentity<F>,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F> + GrindingChallenger<Witness = F>,
{
    pub(crate) fn new(challenger: &'a mut C, shape: FormsSumcheckShape) -> Self {
        let separator = shape.separator::<F, EF>();
        Self {
            state: ProverState::new(challenger, &separator),
            _challenge: PhantomData,
        }
    }

    pub(crate) fn weight(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(PACKING_WEIGHT)
            .into_inner()
    }

    pub(crate) fn round(&mut self, coefficients: [EF; 2]) -> EF {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(COLUMN_ROUND, &coefficients);
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(COLUMN_CHALLENGE)
            .into_inner()
    }

    pub(crate) fn folds(&mut self, folds: &[EF]) {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(FOLDED_COLUMNS, folds);
    }

    pub(crate) fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "forms sumcheck transcript carries no wire values",
        );
    }
}

/// Verifier side of [`FormsSumcheckShape`].
pub(crate) struct FormsSumcheckVerifierTranscript<'a, F: TranscriptField, EF, C> {
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    _challenge: PhantomData<EF>,
}

impl<'a, F, EF, C> FormsSumcheckVerifierTranscript<'a, F, EF, C>
where
    F: Field + TranscriptField + AlgebraIdentity<F>,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F> + GrindingChallenger<Witness = F>,
{
    pub(crate) fn new(challenger: &'a mut C, shape: FormsSumcheckShape) -> Self {
        let separator = shape.separator::<F, EF>();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            _challenge: PhantomData,
        }
    }

    pub(crate) fn weight(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(PACKING_WEIGHT)
            .into_inner()
    }

    pub(crate) fn round(
        &mut self,
        coefficients: [EF; 2],
    ) -> Result<EF, p3_challenger::fs::TranscriptError> {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(COLUMN_ROUND, &coefficients)?;
        Ok(self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(COLUMN_CHALLENGE)
            .into_inner())
    }

    pub(crate) fn folds(&mut self, folds: &[EF]) -> Result<(), p3_challenger::fs::TranscriptError> {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(FOLDED_COLUMNS, folds)?;
        Ok(())
    }

    pub(crate) fn abort(&mut self) {
        self.state.abort();
    }

    pub(crate) fn finish(self) {
        self.state
            .finalize()
            .expect("forms sumcheck transcript is complete");
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;

    use p3_binary_field::BinaryField128;
    use p3_challenger::testing::{SeedDigest, assert_seeds_pairwise_distinct, seed_digest};

    use super::*;

    type EF = BinaryField128;

    /// Two batches over a table of 2^8 rows and three columns, read at the chosen views.
    const fn shape(next: bool) -> ColumnBatchShape {
        ColumnBatchShape {
            table_variables: 8,
            width: 3,
            num_batches: 2,
            next,
            forms: 0,
        }
    }

    /// One batch over a table of 2^9 rows and five columns, read at the chosen views.
    const fn wider_shape(next: bool) -> ColumnBatchShape {
        ColumnBatchShape {
            table_variables: 9,
            width: 5,
            num_batches: 1,
            next,
            forms: 0,
        }
    }

    /// The labels the batches of one shape play, in order.
    fn labels(shape: ColumnBatchShape) -> Vec<&'static str> {
        shape
            .separator::<EF, EF>()
            .pattern()
            .interactions()
            .iter()
            .map(Interaction::label)
            .collect()
    }

    /// The whole byte stream one shape seeds its sponge with, as a comparable digest.
    fn seed_of(shape: ColumnBatchShape) -> SeedDigest {
        seed_digest(&shape.separator::<EF, EF>())
    }

    #[test]
    fn a_current_row_run_is_labelled_by_its_three_numbers() {
        // Invariant: a run reading only the current row binds the shape numbers and nothing
        // else, and each of its batches sends a point and its values before drawing.
        let separator = shape(false).separator::<EF, EF>();
        let mut expected = Vec::new();
        for number in [8u64, 3, 2] {
            expected.extend_from_slice(&8u32.to_be_bytes());
            expected.extend_from_slice(&number.to_be_bytes());
        }
        assert_eq!(separator.instance_label(), expected.as_slice());
        assert_eq!(
            labels(shape(false)),
            [POINT, VALUES, COLUMN_POINT, POINT, VALUES, COLUMN_POINT]
        );
    }

    #[test]
    fn a_current_row_run_seeds_the_bytes_its_literals_name() {
        // Invariant: the whole seed of a current-row run is pinned, not only the instance
        // numbers and the step order.
        //
        //     2^8 rows, width 3, two batches
        //     2^9 rows, width 5, one batch
        //
        // Both halves of the seed reach the digest, so a literal moves exactly when the
        // seeded bytes move. Nothing that leaves the current-row protocol alone does that.
        assert_eq!(
            format!("{:?}", seed_of(shape(false))),
            "2043da134342cdc72a65bfcdb8cd9f7eff45bbeb127fbdd4793f39c23bea23dd",
        );
        assert_eq!(
            format!("{:?}", seed_of(wider_shape(false))),
            "90b5362da1b29b2acd186db9b8ad1cdea5f61298006f88e48aec7beb4ef06ce9",
        );
    }

    #[test]
    fn a_successor_run_seeds_a_transcript_of_its_own() {
        // Invariant: the two runs differ in both halves of the seed, so neither replays the
        // other's challenges, and the successor values are sent before the point is drawn.
        let current = shape(false).separator::<EF, EF>();
        let successor = shape(true).separator::<EF, EF>();
        assert_ne!(current.instance_label(), successor.instance_label());
        assert_ne!(
            current.pattern().pattern_hash(),
            successor.pattern().pattern_hash()
        );
        assert_eq!(
            labels(shape(true)),
            [
                POINT,
                VALUES,
                NEXT_VALUES,
                COLUMN_POINT,
                POINT,
                VALUES,
                NEXT_VALUES,
                COLUMN_POINT
            ]
        );

        // No two of the four runs share a seed, so none of them replays another's draws.
        assert_seeds_pairwise_distinct(&[
            ("current", seed_of(shape(false))),
            ("successor", seed_of(shape(true))),
            ("wider current", seed_of(wider_shape(false))),
            ("wider successor", seed_of(wider_shape(true))),
        ]);
    }
}
