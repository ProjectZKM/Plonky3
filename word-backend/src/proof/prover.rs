//! Proving side of the ordered sub-reductions.
//!
//! # Step order
//!
//! ```text
//!     commitment -> statement
//!                -> integer multiplication: row point, both product trees, product claims
//!                -> vanishing point -> batching coefficient
//!                -> batched zerocheck: zero, AND, low bit, product claims -> operand claims
//!                -> shift batching, public share subtracted
//!                -> bit sumcheck -> word sumcheck -> trace evaluation
//!                -> commitment opening
//! ```
//!
//! The order is load-bearing and the verifier must not diverge from it.
//!
//! One zerocheck covers every local family and the product claims under one coefficient.
//!
//! Binary-field multiplication is not a relation of this language, so no step proves it.
//!
//! # Why the order is sound
//!
//! Every step binds its claims before the challenge that consumes them.
//!
//! - The commitment is bound before the vanishing point, so the trace cannot follow it.
//! - Public words and every dimension are bound before the multiplication row point.
//! - Every product claim is bound before the vanishing point and coefficient that weight it.
//! - The four operand evaluations are bound before the shift reduction draws its batching axes.
//! - The reduced trace value is bound before the commitment draws its opening challenges.
//!
//! Public words enter the sponge as fixed statement data, which is the public check.
//!
//! The shift reduction then subtracts their exact contribution instead of opening them.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::Gf2;
use p3_binary_pcs::{BitOpening, BooleanMultilinearPcs};
use p3_multilinear_util::point::Point;
use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{Algebra, ExtensionField};
use p3_maybe_rayon::prelude::*;
use p3_sumcheck::generic_degree::RoundProver;
use tracing::info_span;

use super::error::WordProofError;
use super::key::WordProofKey;
use super::record::WordProof;
use super::relation::{
    RelationWords, RelationZerocheck, ZEROCHECK_DEGREE, claimed_sum, relation_point,
};
use super::transcript::ProofProverTranscript;
use crate::device::{BITWISE_FAMILY, LINEAR_FAMILY, PRODUCT_FAMILY, WordDevice, is_device_field};
use crate::ports::PortWitness;
use crate::shift::ShiftHints;
use crate::{OperationColumns, Packed, PackedWitness, PackedWord};

impl<W: PackedWord> WordProofKey<W> {
    /// The commitment to the routing words a verifier of this key checks port openings
    /// against, when the key routes ports: computed once per key and commitment scheme.
    ///
    /// # Errors
    ///
    /// Returns an error when the commitment refuses the routing words.
    pub fn port_commitment<EF, Pcs, Challenger>(
        &self,
        pcs: &Pcs,
        challenger: &mut Challenger,
    ) -> Result<Option<Pcs::Commitment>, Pcs::Error>
    where
        Pcs: BooleanMultilinearPcs<EF, Challenger>,
    {
        self.routing_bits(pcs.num_variables())
            .map(|bits| pcs.commit_bits(&bits, challenger).map(|(commitment, _)| commitment))
            .transpose()
    }

    /// [`Self::port_commitment`] with the commitment's prover data, which every proof of the
    /// key opens through [`Self::prove_routed`].
    ///
    /// # Errors
    ///
    /// As [`Self::port_commitment`].
    #[allow(
        clippy::type_complexity,
        reason = "an alias would need bounds it cannot carry"
    )]
    pub fn port_routing<EF, Pcs, Challenger>(
        &self,
        pcs: &Pcs,
        challenger: &mut Challenger,
    ) -> Result<Option<(Pcs::Commitment, Pcs::ProverData)>, Pcs::Error>
    where
        Pcs: BooleanMultilinearPcs<EF, Challenger>,
    {
        self.routing_bits(pcs.num_variables())
            .map(|bits| pcs.commit_bits(&bits, challenger))
            .transpose()
    }

    /// [`Self::prove_with_hints`] for a key that routes ports, against `routing`, the
    /// commitment [`Self::port_routing`] made and its prover data, on `devices` when given as
    /// [`Self::prove_with_hints_on`] runs.
    ///
    /// # Errors
    ///
    /// As [`Self::prove`].
    ///
    /// # Panics
    ///
    /// On a device, unless both fields are the device's, `GF(2^128)`.
    #[allow(
        clippy::type_complexity,
        reason = "an alias would need bounds it cannot carry"
    )]
    pub fn prove_routed<F, EF, Pcs, Challenger>(
        &self,
        pcs: &Pcs,
        values: &PackedWitness<W>,
        routing: (&Pcs::Commitment, &Pcs::ProverData),
        devices: Option<(&dyn WordDevice, &Pcs::OpeningDevice)>,
        challenger: &mut Challenger,
    ) -> Result<
        (
            Pcs::Commitment,
            WordProof<F, EF, Pcs::Proof>,
            ShiftHints<EF>,
        ),
        WordProofError<Pcs::Error>,
    >
    where
        F: TranscriptField,
        EF: ExtensionField<F> + Algebra<Gf2>,
        Pcs: BooleanMultilinearPcs<EF, Challenger>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        if devices.is_some() {
            assert!(
                is_device_field::<F>() && is_device_field::<EF>(),
                "a word device proves over its own field"
            );
        }
        self.prove_by(pcs, values, devices, Some(routing), challenger)
    }

    /// Proves every declared relation and discharges the result through the commitment.
    ///
    /// The commitment is produced here so that it precedes every sampled challenge.
    ///
    /// # Errors
    ///
    /// Returns an error when any of the following holds.
    ///
    /// - The witness has the wrong shape, or the commitment is too narrow for it.
    /// - The statement declares products the challenge field is too small to lift.
    /// - The sampled batching coefficient vanishes.
    /// - The commitment refuses the trace or its opening.
    #[allow(
        clippy::type_complexity,
        reason = "an alias would need bounds it cannot carry"
    )]
    pub fn prove<F, EF, Pcs, Challenger>(
        &self,
        pcs: &Pcs,
        values: &PackedWitness<W>,
        challenger: &mut Challenger,
    ) -> Result<(Pcs::Commitment, WordProof<F, EF, Pcs::Proof>), WordProofError<Pcs::Error>>
    where
        F: TranscriptField,
        EF: ExtensionField<F> + Algebra<Gf2>,
        Pcs: BooleanMultilinearPcs<EF, Challenger>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let (commitment, proof, _) = self.prove_with_hints(pcs, values, challenger)?;
        Ok((commitment, proof))
    }

    /// [`Self::prove`], also returning the values a verifier may defer: see
    /// [`ShiftHints`].
    ///
    /// # Errors
    ///
    /// As [`Self::prove`].
    #[allow(
        clippy::type_complexity,
        reason = "an alias would need bounds it cannot carry"
    )]
    pub fn prove_with_hints<F, EF, Pcs, Challenger>(
        &self,
        pcs: &Pcs,
        values: &PackedWitness<W>,
        challenger: &mut Challenger,
    ) -> Result<
        (
            Pcs::Commitment,
            WordProof<F, EF, Pcs::Proof>,
            ShiftHints<EF>,
        ),
        WordProofError<Pcs::Error>,
    >
    where
        F: TranscriptField,
        EF: ExtensionField<F> + Algebra<Gf2>,
        Pcs: BooleanMultilinearPcs<EF, Challenger>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        self.prove_by(pcs, values, None, None, challenger)
    }

    /// [`Self::prove_with_hints`] with the product trees, the selector, the relation check and
    /// the shift reduction on `device`, and the commitment's opening on `opening`.
    ///
    /// The proof is the one [`Self::prove_with_hints`] makes.
    ///
    /// # Errors
    ///
    /// As [`Self::prove`].
    ///
    /// # Panics
    ///
    /// Unless both fields are the device's, `GF(2^128)`.
    #[allow(
        clippy::type_complexity,
        reason = "an alias would need bounds it cannot carry"
    )]
    pub fn prove_with_hints_on<F, EF, Pcs, Challenger>(
        &self,
        pcs: &Pcs,
        values: &PackedWitness<W>,
        device: &dyn WordDevice,
        opening: &Pcs::OpeningDevice,
        challenger: &mut Challenger,
    ) -> Result<
        (
            Pcs::Commitment,
            WordProof<F, EF, Pcs::Proof>,
            ShiftHints<EF>,
        ),
        WordProofError<Pcs::Error>,
    >
    where
        F: TranscriptField,
        EF: ExtensionField<F> + Algebra<Gf2>,
        Pcs: BooleanMultilinearPcs<EF, Challenger>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        assert!(
            is_device_field::<F>() && is_device_field::<EF>(),
            "a word device proves over its own field"
        );
        self.prove_by(pcs, values, Some((device, opening)), None, challenger)
    }

    #[allow(
        clippy::type_complexity,
        reason = "an alias would need bounds it cannot carry"
    )]
    fn prove_by<F, EF, Pcs, Challenger>(
        &self,
        pcs: &Pcs,
        values: &PackedWitness<W>,
        devices: Option<(&dyn WordDevice, &Pcs::OpeningDevice)>,
        routing: Option<(&Pcs::Commitment, &Pcs::ProverData)>,
        challenger: &mut Challenger,
    ) -> Result<
        (
            Pcs::Commitment,
            WordProof<F, EF, Pcs::Proof>,
            ShiftHints<EF>,
        ),
        WordProofError<Pcs::Error>,
    >
    where
        F: TranscriptField,
        EF: ExtensionField<F> + Algebra<Gf2>,
        Pcs: BooleanMultilinearPcs<EF, Challenger>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Reject every shape before the transcript absorbs a single statement value.
        if devices.is_some() && self.ports.is_some() && routing.is_none() {
            return Err(WordProofError::DevicePorts);
        }
        let device = devices.map(|(device, _)| device);
        let commitment_variables = pcs.num_variables();
        self.validate_arity(commitment_variables)?;
        let shape = |error: crate::witness::WitnessError| WordProofError::SegmentLength {
            segment: error.segment,
            expected: error.expected,
            actual: error.actual,
        };
        values.check_shape(self.statement()).map_err(shape)?;

        if let Some(reduction) = &self.integer_mul {
            reduction.check_field::<EF>()?;
        }

        // Binding the commitment first stops the trace being chosen after the challenges.
        // A device opening a routed key's pair holds the committed words, so it commits them.
        let (commitment, prover_data) = info_span!("commit")
            .in_scope(|| match (devices, &self.ports) {
                (Some((_, opening)), Some(_)) => {
                    assert_eq!(W::BITS, 64, "a device holds 64-bit words");
                    let words = values.witness_len();
                    assert!(
                        words <= 1 << (commitment_variables - Self::bit_variables()),
                        "the witness fits the commitment"
                    );
                    pcs.commit_held_on(words, opening, challenger)
                }
                (Some((_, opening)), None) => {
                    pcs.commit_bits_on(&self.trace_bits(values, commitment_variables), opening, challenger)
                }
                (None, _) => pcs.commit_bits(&self.trace_bits(values, commitment_variables), challenger),
            })
            .map_err(WordProofError::Commitment)?;

        let committed_routing;
        let routing = match (&self.ports, routing) {
            (None, _) => None,
            (Some(_), Some((commitment, data))) => {
                pcs.observe_commitment(commitment, challenger);
                Some(data)
            }
            (Some(_), None) => {
                let bits = self
                    .routing_bits(commitment_variables)
                    .expect("a key that routes ports has routing words");
                committed_routing = info_span!("commit routing")
                    .in_scope(|| pcs.commit_bits(&bits, challenger))
                    .map_err(WordProofError::Commitment)?
                    .1;
                Some(&committed_routing)
            }
        };

        // A device holding the committed words evaluates each family's columns where they stay,
        // the product columns now and the others once the product reduction is done.
        let public_u64 = values
            .public()
            .iter()
            .map(|&packed| W::unpack(packed).to_u64())
            .collect::<Vec<_>>();
        let evaluate = |device: &dyn WordDevice, families: u32| {
            info_span!("device operation columns", families).in_scope(|| {
                device.operation_columns(
                    values.witness_len(),
                    &public_u64,
                    self.shift.device_key(),
                    &|| self.shift.device_layout(false),
                    &|| self.shift.device_layout(true),
                    1 << self.shift.constraint_variables(),
                    families,
                )
            })
        };
        let product_family = if self.integer_mul.is_some() { PRODUCT_FAMILY } else { 0 };
        let resident = device.is_some_and(|device| evaluate(device, product_family));
        let columns = if resident {
            None
        } else {
            let columns = info_span!("operation columns")
                .in_scope(|| OperationColumns::new(self.statement(), values));
            Some(columns.map_err(shape)?)
        };

        // Both relation challenges follow the complete statement.
        let public_words = values
            .public()
            .iter()
            .copied()
            .map(W::unpack)
            .collect::<Vec<_>>();
        let variables = self.zerocheck_variables();
        let mut transcript = ProofProverTranscript::<_, F, EF>::new(
            challenger,
            self.transcript_shape(commitment_variables),
            &public_words,
        );

        // Every product reduces to claims on its four columns before either relation draw.
        let product = transcript.integer_mul(|challenger| {
            info_span!("integer_mul").in_scope(|| {
                self.integer_mul.map(|reduction| {
                    reduction.prove::<F, EF, W, _>(
                        columns.as_ref().map(OperationColumns::integer_mul),
                        device,
                        challenger,
                    )
                })
            })
        });
        let (vanishing_point, batching) = transcript
            .challenges(variables)
            .ok_or(WordProofError::DegenerateBatching)?;

        // The local terms vanish on the padded cube and each claim term sums to its claim.
        let (integer_mul, points, sum) = match product {
            Some((proof, claims)) => {
                let sum = claimed_sum(&claims.each_ref().map(|claim| claim.value), batching);
                let points = self.product_points(&vanishing_point, &claims);
                (Some(proof), Some(points), sum)
            }
            None => (None, None, EF::ZERO),
        };
        let unpack = |column: &[Packed<W>]| -> Vec<u64> {
            column
                .iter()
                .map(|&packed| W::unpack(packed).to_u64())
                .collect()
        };
        let words = info_span!("relation words").in_scope(|| match &columns {
            Some(columns) => RelationWords {
                linear: unpack(columns.zero()),
                bitwise: columns
                    .bitwise_and()
                    .each_ref()
                    .map(|column| unpack(column)),
                product: points.as_ref().map(|_| {
                    columns
                        .integer_mul()
                        .each_ref()
                        .map(|column| unpack(column))
                }),
            },
            None => RelationWords {
                linear: Vec::new(),
                bitwise: core::array::from_fn(|_| Vec::new()),
                product: points.as_ref().map(|_| core::array::from_fn(|_| Vec::new())),
            },
        });
        if let (true, Some(device)) = (resident, device) {
            assert!(
                evaluate(device, LINEAR_FAMILY | BITWISE_FAMILY),
                "the device still holds the committed words"
            );
        }
        let span = info_span!("zerocheck").entered();
        let mut prover = RelationZerocheck::new(
            &vanishing_point,
            self.shift.constraint_variables(),
            Self::bit_variables(),
            words,
            points,
            batching,
            device,
            resident,
        );
        let (zerocheck, point) = transcript.zerocheck(|challenger| {
            prover.prove::<F, _>(challenger, variables, ZEROCHECK_DEGREE, 0, sum)
        });
        drop(span);
        let point = relation_point(&point, Self::bit_variables());
        let operands = prover.terminal_operands();
        drop(prover);
        transcript.finish();

        // Every operand claim shares the point the vanishing check ended on.
        let claim = self.operand_claim(&point, &operands);
        let (shift, opening, hints) = info_span!("shift").in_scope(|| {
            self.shift.prove_by::<F, EF, _>(
                values,
                &claim,
                device,
                resident,
                self.ports.is_some(),
                challenger,
            )
        })?;

        let ports = match &self.ports {
            Some(routing) => Some(info_span!("ports").in_scope(|| {
                let words;
                let witness = match (device, resident) {
                    (Some(device), true) => PortWitness::Device(device, values.witness_len()),
                    _ => {
                        words = values
                            .witness()
                            .par_iter()
                            .map(|&packed| W::unpack(packed).to_u64())
                            .collect::<Vec<_>>();
                        PortWitness::Host(&words)
                    }
                };
                routing
                    .prove::<F, EF, _>(
                        witness,
                        self.shift.word_variables(),
                        claim.bit_point(),
                        challenger,
                    )
                    .map_err(WordProofError::Ports)
            })?),
            None => None,
        };

        // One authenticated opening closes the whole protocol.
        let mut points = vec![self.commitment_point(opening.point(), commitment_variables)];
        let (opened, opening_proof, port_opening) = match (routing, &ports) {
            (Some(routing_data), Some((_, claims))) => {
                points.push(self.commitment_point(&claims.trace.0, commitment_variables));
                let routing_point = [self.commitment_point(&claims.routing.0, commitment_variables)];
                let ((trace, trace_proof), (routing, routing_proof)) = info_span!("open pair")
                    .in_scope(|| {
                        let first = (prover_data, current_openings(&points));
                        let second = (routing_data, current_openings(&routing_point));
                        match devices {
                            Some((_, device)) => pcs.open_readings_pair_on(
                                pcs,
                                (first.0, &first.1),
                                (second.0, &second.1),
                                device,
                                challenger,
                            ),
                            None => pcs.open_readings_pair(
                                pcs,
                                (first.0, &first.1),
                                (second.0, &second.1),
                                challenger,
                            ),
                        }
                    })
                    .map_err(WordProofError::Commitment)?;
                if routing.first().and_then(|reading| reading.current) != Some(claims.routing.1) {
                    return Err(WordProofError::OpeningValue);
                }
                let opened = trace.iter().filter_map(|reading| reading.current).collect();
                (opened, trace_proof, Some(routing_proof))
            }
            _ => {
                let (opened, proof) = info_span!("open")
                    .in_scope(|| match devices {
                        Some((_, device)) => {
                            pcs.open_at_points_on(prover_data, &points, device, challenger)
                        }
                        None => pcs.open_at_points(prover_data, &points, challenger),
                    })
                    .map_err(WordProofError::Commitment)?;
                (opened, proof, None)
            }
        };
        let opened: Vec<EF> = opened;
        if opened.first() != Some(&opening.value()) {
            return Err(WordProofError::OpeningValue);
        }
        if let Some((_, claims)) = &ports
            && opened.get(1) != Some(&claims.trace.1)
        {
            return Err(WordProofError::OpeningValue);
        }

        Ok((
            commitment,
            WordProof {
                integer_mul,
                zerocheck,
                operands,
                shift,
                ports: ports.map(|(proof, _)| proof),
                opening: opening_proof,
                port_opening,
            },
            hints,
        ))
    }
}

/// Each point as an opening asking for the witness there alone.
pub(super) fn current_openings<EF: Clone>(points: &[Point<EF>]) -> Vec<BitOpening<EF>> {
    points
        .iter()
        .map(|point| BitOpening {
            point: point.clone(),
            row_variables: point.num_variables(),
            current: true,
            next: false,
        })
        .collect()
}
