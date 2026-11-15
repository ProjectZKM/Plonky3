//! Typed transcript fixing the order of every multiplication-reduction interaction.
//!
//! ```text
//!     row point -> shared root
//!               -> power layers (sumcheck, halves, line) x k -> power leaves -> fold point
//!               -> selector sumcheck -> base lift and right bits -> right point
//!               -> base layers (sumcheck, halves, line) x k -> left bits -> left point
//!               -> result layers (sumcheck, halves, line) x (k + 1) -> limb bits -> limb point
//! ```
//!
//! Every value the prover sends is absorbed before the challenge that consumes it.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample};
use p3_field::ExtensionField;

use super::Tree;

/// Version byte bound into the protocol seed.
const VERSION: u8 = 1;

/// Protocol name bound into the protocol seed.
const NAME: &[u8] = b"p3-word-integer-mul";

/// Label of the row point both lifts are compared at.
const ROW_POINT: &str = "row_point";

/// Label of the shared value of both lifts at the row point.
const ROOT: &str = "root";

/// Label of the two halves one layer sumcheck ends on.
const HALVES: &str = "halves";

/// Label of the challenge joining two halves into one claim.
const LINE: &str = "line";

/// Label of the base lift and right bits the selector sumcheck ends on.
const SELECTED: &str = "selected";

/// Label of the point folding the right bits into one claim.
const RIGHT_POINT: &str = "right_point";

/// Type-level identity of one layer sumcheck.
struct Layer;

/// Type-level identity of the selector sumcheck.
struct Selector;

/// Label of the selector sumcheck.
const SELECTOR: &str = "selector";

/// Sponge alphabet of a challenger native to the challenge field.
type Alphabet<F> = FieldUnit<F>;

/// Dimensions that fix every multiplication-reduction interaction.
#[derive(Clone, Copy)]
pub(super) struct TranscriptShape {
    /// Number of padded multiplication-row variables.
    pub(super) row_variables: usize,
    /// Number of within-word variables.
    pub(super) bit_variables: usize,
}

impl TranscriptShape {
    /// Describes the row draw, the root, and both trees in order.
    fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let message = |label, count| {
            Interaction::algebra::<F, EF>(Hierarchy::Atomic, Kind::Message, label, count)
        };
        let challenge = |label, count| {
            Interaction::algebra::<F, EF>(Hierarchy::Atomic, Kind::Challenge, label, count)
        };

        let mut steps = vec![
            challenge(ROW_POINT, Length::Fixed(self.row_variables)),
            message(ROOT, Length::Scalar),
        ];
        let layers = |steps: &mut Vec<Interaction>, tree: Tree| {
            for _ in 0..tree.depth(self.bit_variables) {
                steps.push(Interaction::marker::<Layer>(
                    Hierarchy::Begin,
                    Kind::Protocol,
                    tree.layer_label(),
                ));
                steps.push(Interaction::marker::<Layer>(
                    Hierarchy::End,
                    Kind::Protocol,
                    tree.layer_label(),
                ));
                steps.push(message(HALVES, Length::Fixed(2)));
                steps.push(challenge(LINE, Length::Scalar));
            }
            steps.push(message(
                tree.leaves_label(),
                Length::Fixed(tree.leaves(self.bit_variables)),
            ));
            steps.push(challenge(
                tree.point_label(),
                Length::Fixed(self.bit_variables),
            ));
        };
        layers(&mut steps, Tree::Power);
        steps.push(Interaction::marker::<Selector>(
            Hierarchy::Begin,
            Kind::Protocol,
            SELECTOR,
        ));
        steps.push(Interaction::marker::<Selector>(
            Hierarchy::End,
            Kind::Protocol,
            SELECTOR,
        ));
        steps.push(message(
            SELECTED,
            Length::Fixed((1 << self.bit_variables) + 1),
        ));
        steps.push(challenge(RIGHT_POINT, Length::Fixed(self.bit_variables)));
        layers(&mut steps, Tree::Base);
        layers(&mut steps, Tree::Result);
        InteractionPattern::new(steps)
            .expect("balanced brackets around atomic steps are well formed")
    }

    /// Binds the protocol identity and both dimensions.
    fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());
        for dimension in [self.row_variables, self.bit_variables] {
            separator.instance(&(dimension as u64).to_le_bytes());
        }
        separator
    }
}

/// Prover-side driver for the multiplication reduction.
pub(super) struct ProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Pattern player and borrowed challenger.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// Marker for the sampled field.
    _field: PhantomData<EF>,
}

impl<'a, C, F, EF> ProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seeds the reduction from its dimensions.
    pub(super) fn new(challenger: &'a mut C, shape: TranscriptShape) -> Self {
        Self {
            state: ProverState::new(challenger, &shape.domain_separator::<F, EF>()),
            _field: PhantomData,
        }
    }

    /// Draws the row point, then binds the root the prover computes at it.
    pub(super) fn row_point(
        &mut self,
        count: usize,
        root: impl FnOnce(&[EF]) -> EF,
    ) -> (Vec<EF>, EF) {
        let point = self
            .state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(ROW_POINT, count)
            .into_iter()
            .map(|value| value.into_inner())
            .collect::<Vec<_>>();
        let value = root(&point);
        self.state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(ROOT, &value);
        (point, value)
    }

    /// Lends the challenger to one layer sumcheck, then binds its halves and draws the line.
    pub(super) fn layer<R>(
        &mut self,
        tree: Tree,
        run: impl FnOnce(&mut &'a mut C) -> (R, [EF; 2]),
    ) -> (R, [EF; 2], EF) {
        self.state.begin_protocol::<Layer>(tree.layer_label());
        let (result, halves) = run(self.state.challenger_mut());
        self.state.end_protocol::<Layer>(tree.layer_label());
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(HALVES, &halves);
        let line = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(LINE)
            .into_inner();
        (result, halves, line)
    }

    /// Binds one tree's leaf evaluations, then draws the point folding them.
    pub(super) fn leaves(&mut self, tree: Tree, values: &[EF], count: usize) -> Vec<EF> {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(tree.leaves_label(), values);
        self.points(tree.point_label(), count)
    }

    /// Lends the challenger to the selector sumcheck, then binds its values and draws a point.
    pub(super) fn selector<R>(
        &mut self,
        count: usize,
        run: impl FnOnce(&mut &'a mut C) -> (R, Vec<EF>),
    ) -> (R, Vec<EF>, Vec<EF>) {
        self.state.begin_protocol::<Selector>(SELECTOR);
        let (result, values) = run(self.state.challenger_mut());
        self.state.end_protocol::<Selector>(SELECTOR);
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(SELECTED, &values);
        let point = self.points(RIGHT_POINT, count);
        (result, values, point)
    }

    /// Draws `count` challenges under one label.
    fn points(&mut self, label: &'static str, count: usize) -> Vec<EF> {
        self.state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(label, count)
            .into_iter()
            .map(|value| value.into_inner())
            .collect()
    }

    /// Closes the reduction transcript.
    pub(super) fn finish(self) {
        // Every value is carried by the record, so nothing was written.
        assert!(self.state.finalize().is_empty());
    }
}

/// Verifier-side replay of the multiplication reduction.
pub(super) struct VerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Pattern player over an empty wire and a borrowed challenger.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// Marker for the sampled field.
    _field: PhantomData<EF>,
}

impl<'a, C, F, EF> VerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Reseeds the reduction from the same dimensions.
    pub(super) fn new(challenger: &'a mut C, shape: TranscriptShape) -> Self {
        Self {
            state: VerifierState::new(challenger, &shape.domain_separator::<F, EF>(), &[]),
            _field: PhantomData,
        }
    }

    /// Replays the row draw, then binds the root the record carries.
    pub(super) fn row_point(&mut self, count: usize, root: EF) -> Vec<EF> {
        let point = self
            .state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(ROW_POINT, count)
            .into_iter()
            .map(|value| value.into_inner())
            .collect();
        self.state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(ROOT, &root);
        point
    }

    /// Lends the challenger to one layer replay, then binds its halves and draws the line.
    pub(super) fn layer<R>(
        &mut self,
        tree: Tree,
        halves: &[EF; 2],
        run: impl FnOnce(&mut &'a mut C) -> R,
    ) -> (R, EF) {
        self.state.begin_protocol::<Layer>(tree.layer_label());
        let result = run(self.state.challenger_mut());
        self.state.end_protocol::<Layer>(tree.layer_label());
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(HALVES, halves)
            .expect("a fixed-width array matches its fixed-width step");
        let line = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(LINE)
            .into_inner();
        (result, line)
    }

    /// Binds one tree's leaf evaluations, then draws the point folding them.
    ///
    /// # Errors
    ///
    /// Returns `None` when the record carries a different number of leaves than the step.
    pub(super) fn leaves(&mut self, tree: Tree, values: &[EF], count: usize) -> Option<Vec<EF>> {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(tree.leaves_label(), values)
            .ok()?;
        Some(self.points(tree.point_label(), count))
    }

    /// Lends the challenger to the selector replay, then binds its values and draws a point.
    pub(super) fn selector<R>(
        &mut self,
        values: &[EF],
        count: usize,
        run: impl FnOnce(&mut &'a mut C) -> R,
    ) -> Option<(R, Vec<EF>)> {
        self.state.begin_protocol::<Selector>(SELECTOR);
        let result = run(self.state.challenger_mut());
        self.state.end_protocol::<Selector>(SELECTOR);
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(SELECTED, values)
            .ok()?;
        Some((result, self.points(RIGHT_POINT, count)))
    }

    /// Draws `count` challenges under one label.
    fn points(&mut self, label: &'static str, count: usize) -> Vec<EF> {
        self.state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(label, count)
            .into_iter()
            .map(|value| value.into_inner())
            .collect()
    }

    /// Closes the reduction transcript.
    pub(super) fn finish(self) {
        self.state
            .finalize()
            .expect("the multiplication reduction reads an empty wire");
    }

    /// Releases completeness checks after a rejected record.
    pub(super) fn abort(&mut self) {
        // A malformed record may stop the replay part-way through the pattern.
        self.state.abort();
    }
}
