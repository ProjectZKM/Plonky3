//! Typed transcript for the copy constraints between committed ports.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample};
use p3_field::{BasedVectorSpace, ExtensionField, PrimeCharacteristicRing};

/// Version byte bound into the protocol seed.
const VERSION: u8 = 2;

/// Protocol name bound into the protocol seed.
const NAME: &[u8] = b"p3-word-port-routing";

/// Label of the within-word point the port values are read at.
const BIT_POINT: &str = "bit_point";

/// Label of the fingerprint offset.
const OFFSET: &str = "offset";

/// Label of the label coefficient.
const LABEL: &str = "label";

/// Label of the point the cell labels are read at.
const LABEL_POINT: &str = "label_point";

/// Label around the product reduction.
const PRODUCT: &str = "product";

/// Label of the permuted labels' claimed evaluation.
const PERMUTED_LABEL: &str = "permuted_label";

/// Type-level identity of the product sub-protocol.
struct Product;

/// Sponge alphabet of a challenger native to the challenge field.
type Alphabet<F> = FieldUnit<F>;

/// Routing dimensions that fix every transcript interaction.
#[derive(Clone, Copy)]
pub(super) struct PortShape {
    /// Number of within-word variables.
    pub(super) bit_variables: usize,
    /// Number of variables of the label point.
    pub(super) label_variables: usize,
    /// Number of committed-word variables, the cells' index variables.
    pub(super) word_variables: usize,
    /// Number of cells the routing moves.
    pub(super) moved: usize,
}

impl PortShape {
    fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let steps = vec![
            Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Public,
                BIT_POINT,
                Length::Fixed(self.bit_variables * EF::DIMENSION),
            ),
            Interaction::algebra::<F, EF>(Hierarchy::Atomic, Kind::Challenge, OFFSET, Length::Fixed(1)),
            Interaction::algebra::<F, EF>(Hierarchy::Atomic, Kind::Challenge, LABEL, Length::Fixed(1)),
            Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                LABEL_POINT,
                Length::Fixed(self.label_variables),
            ),
            Interaction::marker::<Product>(Hierarchy::Begin, Kind::Protocol, PRODUCT),
            Interaction::marker::<Product>(Hierarchy::End, Kind::Protocol, PRODUCT),
            Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Public,
                PERMUTED_LABEL,
                Length::Fixed(EF::DIMENSION),
            ),
        ];
        InteractionPattern::new(steps).expect("a flat sequence of atomic steps is well formed")
    }

    fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());
        for dimension in [self.bit_variables, self.label_variables, self.word_variables, self.moved] {
            separator.instance(&(dimension as u64).to_le_bytes());
        }
        separator
    }
}

/// The three fingerprint challenges.
pub(super) struct PortChallenges<EF> {
    /// The offset every fingerprint carries.
    pub(super) offset: EF,
    /// The coefficient of the cell label.
    pub(super) label: EF,
    /// The point the label bits are combined at.
    pub(super) label_point: Vec<EF>,
}

/// Prover-side driver.
pub(super) struct PortProverTranscript<'a, C, F: TranscriptField, EF> {
    state: ProverState<&'a mut C, Alphabet<F>>,
    _field: PhantomData<EF>,
}

impl<'a, C, F, EF> PortProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Binds the routing dimensions and the within-word point.
    pub(super) fn new(challenger: &'a mut C, shape: PortShape, bit_point: &[EF]) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        let mut state = ProverState::new(challenger, &separator);
        state.add_public_scalars::<F, FieldToFieldCodec<F>>(BIT_POINT, &flatten_extensions(bit_point));
        Self {
            state,
            _field: PhantomData,
        }
    }

    /// Samples the offset, the label coefficient, then the label point.
    pub(super) fn challenges(&mut self, label_variables: usize) -> PortChallenges<EF> {
        let mut draw = |label, count| {
            self.state
                .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(label, count)
                .into_iter()
                .map(|value| value.into_inner())
                .collect::<Vec<_>>()
        };
        let offset = draw(OFFSET, 1)[0];
        let label = draw(LABEL, 1)[0];
        let label_point = draw(LABEL_POINT, label_variables);
        PortChallenges {
            offset,
            label,
            label_point,
        }
    }

    /// Lends the challenger to the product reduction.
    pub(super) fn product<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<Product>(PRODUCT);
        let result = run(self.state.challenger_mut());
        self.state.end_protocol::<Product>(PRODUCT);
        result
    }

    /// Absorbs the permuted labels' claimed evaluation, then closes the transcript.
    pub(super) fn permuted_label(mut self, value: EF) {
        self.state
            .add_public_scalars::<F, FieldToFieldCodec<F>>(PERMUTED_LABEL, &flatten_extensions(&[value]));
        assert!(self.state.finalize().is_empty());
    }
}

/// Verifier-side replay.
pub(super) struct PortVerifierTranscript<'a, C, F: TranscriptField, EF> {
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    _field: PhantomData<EF>,
}

impl<'a, C, F, EF> PortVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Reabsorbs the routing dimensions and the within-word point.
    pub(super) fn new(challenger: &'a mut C, shape: PortShape, bit_point: &[EF]) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        let mut state = VerifierState::new(challenger, &separator, &[]);
        state.observe_public_scalars::<F, FieldToFieldCodec<F>>(BIT_POINT, &flatten_extensions(bit_point));
        Self {
            state,
            _field: PhantomData,
        }
    }

    /// Replays the offset, the label coefficient, then the label point.
    pub(super) fn challenges(&mut self, label_variables: usize) -> PortChallenges<EF> {
        let mut draw = |label, count| {
            self.state
                .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(label, count)
                .into_iter()
                .map(|value| value.into_inner())
                .collect::<Vec<_>>()
        };
        let offset = draw(OFFSET, 1)[0];
        let label = draw(LABEL, 1)[0];
        let label_point = draw(LABEL_POINT, label_variables);
        PortChallenges {
            offset,
            label,
            label_point,
        }
    }

    /// Lends the challenger to the product replay.
    pub(super) fn product<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<Product>(PRODUCT);
        let result = run(self.state.challenger_mut());
        self.state.end_protocol::<Product>(PRODUCT);
        result
    }

    /// Reabsorbs the permuted labels' claimed evaluation, then closes the replay.
    pub(super) fn permuted_label(mut self, value: EF) {
        self.state
            .observe_public_scalars::<F, FieldToFieldCodec<F>>(PERMUTED_LABEL, &flatten_extensions(&[value]));
        self.state
            .finalize()
            .expect("the port reduction reads an empty top-level wire");
    }

    /// Releases completeness checks after a rejected delegated proof.
    pub(super) fn abort(&mut self) {
        self.state.abort();
    }
}

/// Flattens extension elements into their canonical base-field coordinates.
fn flatten_extensions<F, EF>(values: &[EF]) -> Vec<F>
where
    F: Copy + PrimeCharacteristicRing,
    EF: BasedVectorSpace<F>,
{
    values
        .iter()
        .flat_map(|value| value.as_basis_coefficients_slice().iter().copied())
        .collect()
}
