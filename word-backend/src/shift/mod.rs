//! Sumcheck reduction from shifted operands to one committed-word opening.
//!
//! The reduction proves one identity in two stages:
//!
//! ```text
//! batched operand claim - public share
//!     = sum_(sequence, bit) sparse_witness(sequence, bit) * shift(sequence, bit)
//!     = sum_word committed_word(word, r_bit) * wiring(word, r_bit).
//! ```
//!
//! The first sumcheck binds the 32-bit or 64-bit coordinate.
//!
//! The second binds the committed word index.
//!
//! Source words remain bit-packed until the first point has been sampled.
//!
//! Shift spellings are verifier-fixed metadata, so they are summed exactly rather than opened as prover polynomials.
//!
//! A two-slot operator applies the outer transpose first and the inner one second.
//!
//! That ordering preserves composition without adding sampled shift-code variables.
//!
//! A false operand claim either survives as a false final opening or hits a batching or sumcheck collision.
//!
//! Before the PCS opening that algebraic error is `(4 + 2 * (log2(word_bits) + log2(padded_words))) / |EF|`.

mod claim;
mod error;
mod polynomial;
pub(crate) mod transcript;
mod wiring;

use alloc::vec::Vec;
use core::marker::PhantomData;
use core::num::NonZeroUsize;
use core::sync::atomic::{AtomicU64, Ordering};

pub use claim::{ShiftClaim, ShiftOpeningClaim};
pub use error::ShiftReductionError;
use p3_binary_field::Gf2;
use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{Algebra, ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_security::SecurityTerm;
use p3_security::word::WordShiftSecurityModel;
use p3_sumcheck::generic_degree::{GenericDegreeProof, RoundProver};
use p3_word::{Segment, Word};
use serde::{Deserialize, Serialize};
use tracing::info_span;
use transcript::{
    OPERAND_VARIABLES, OPERATION_VARIABLES, ShiftProverTranscript, ShiftVerifierTranscript,
    TranscriptShape,
};
pub(crate) use wiring::{byte_tables, evaluate_bytes};
use wiring::{
    PreparedWeights, bit_pairs, bit_prover, operand_index, public_bit_weights,
    public_contribution, shift_evaluations, wiring_evaluation, word_prover,
};

use crate::device::{DeviceShift, WordDevice, Words, from_device, slice_to_device, to_device};
use crate::statement::Statement;
use crate::{CompiledKeyLayout, KeyCompileError, PackedWitness, PackedWord};

/// The word phase on a device, its factors resident there.
struct DeviceWordPhase<'a, F> {
    shift: &'a mut dyn DeviceShift,
    _field: PhantomData<F>,
}

impl<F: Field> RoundProver<F> for DeviceWordPhase<'_, F> {
    fn fold(&mut self, r: F) {
        self.shift
            .fold(to_device(r).expect("only the device's field reaches the device"));
    }

    fn round_poly(&self) -> Vec<F> {
        self.shift.round().into_iter().map(from_device).collect()
    }
}

/// A value of the shift reduction's verifier that is a function of the key and the transcript
/// alone, which a caller may defer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Deferred {
    /// The share of the public words below the caller's `inputs`, at the claim's points; the
    /// query's sumcheck points are empty.  It is a function of those words, so a caller
    /// evaluates it rather than deferring it: see [`ShiftReductionKey::input_bit_weights`].
    Inputs,
    /// The share of the public words the key fixes, at the claim's points; the query's
    /// sumcheck points are empty.
    Constants,
    /// The wiring factor.
    Wiring,
}

/// The deferred values as the prover knows them: the share of every public word in the
/// batched operand claim, and the wiring factor at the point the reduction ends on.  A verifier
/// that defers them reads them from here instead of evaluating the key; whoever checks the
/// deferred claims evaluates the key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShiftHints<EF> {
    /// [`Deferred::Inputs`] plus [`Deferred::Constants`]: every public word's share.
    pub public: EF,
    /// [`Deferred::Wiring`].
    pub wiring: EF,
}

/// Where the shift reduction's wiring factor is evaluated: the transcript values it depends
/// on.  The factor is a function of the key and these alone.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WiringQuery<F> {
    /// The point selecting a padded constraint row.
    pub constraint_point: Vec<F>,
    /// The relation-family batching weights.
    pub operation: Vec<F>,
    /// The operand-position batching weights.
    pub operand: Vec<F>,
    /// The claim's point selecting a bit within each word.
    pub claim_bit_point: Vec<F>,
    /// The point the bit-index sumcheck ends at.
    pub bit_point: Vec<F>,
    /// The point the word-index sumcheck ends at.
    pub word_point: Vec<F>,
}

impl<F> WiringQuery<F> {
    /// Every value of the query, in field order.
    pub fn values(&self) -> impl Iterator<Item = &F> {
        self.constraint_point
            .iter()
            .chain(&self.operation)
            .chain(&self.operand)
            .chain(&self.claim_bit_point)
            .chain(&self.bit_point)
            .chain(&self.word_point)
    }

    /// The query with every value mapped by `f`.
    pub fn map<G>(&self, mut f: impl FnMut(&F) -> G) -> WiringQuery<G> {
        let mut each = |values: &Vec<F>| values.iter().map(&mut f).collect();
        WiringQuery {
            constraint_point: each(&self.constraint_point),
            operation: each(&self.operation),
            operand: each(&self.operand),
            claim_bit_point: each(&self.claim_bit_point),
            bit_point: each(&self.bit_point),
            word_point: each(&self.word_point),
        }
    }
}

/// Transcript record for the two quadratic sumchecks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShiftReductionProof<F, EF> {
    /// Rounds binding the within-word bit coordinate.
    bit_sumcheck: GenericDegreeProof<F, EF>,
    /// Rounds binding the committed word coordinate.
    word_sumcheck: GenericDegreeProof<F, EF>,
    /// Evaluation claimed for the committed bit trace.
    witness_evaluation: EF,
}

/// A checked word statement together with its compiled shift wiring.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ShiftReductionKey<W: Word> {
    /// The relations defining the statement, flat or composed.
    statement: Statement<W>,
    /// Sparse word-to-relation metadata used by both proving phases.
    layout: CompiledKeyLayout<W>,
    /// Variables spanning the widest padded relation family.
    constraint_variables: usize,
    /// The identity a device keeps this key's wiring under; a loaded key takes a fresh one.
    #[serde(skip, default = "KeyIdentity::fresh")]
    identity: KeyIdentity,
}

/// A key's identity, shared by its clones, which wire alike; it plays no part in equality.
#[derive(Clone, Copy, Debug)]
struct KeyIdentity(u64);

impl KeyIdentity {
    fn fresh() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

impl PartialEq for KeyIdentity {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for KeyIdentity {}

impl<W: Word> ShiftReductionKey<W> {
    /// The identity a device keeps this key's wiring under.
    pub(crate) const fn device_key(&self) -> u64 {
        self.identity.0
    }

    /// The committed words' wiring flattened for a device, or the public words' with `public`.
    pub(crate) fn device_layout(&self, public: bool) -> crate::ShiftLayout {
        let segment = if public {
            self.layout.public()
        } else {
            self.layout.witness()
        };
        segment.shift_layout(operand_index)
    }

    /// Compiles a checked statement into a reusable reduction key.
    ///
    /// A composed statement compiles each gadget once, whatever its instance count.
    ///
    /// It reaches the same wiring the lowered flat statement would.
    ///
    /// # Errors
    ///
    /// Returns an error when the statement outgrows the compact key representation.
    pub fn new(statement: impl Into<Statement<W>>) -> Result<Self, KeyCompileError> {
        // Key compilation fixes every sparse reference before proving begins.
        let statement = statement.into();
        let layout = statement.compiled_layout()?;
        let constraint_count = statement
            .relation_counts()
            .into_iter()
            .max()
            .unwrap_or(0)
            .max(1);
        let constraint_variables = constraint_count.next_power_of_two().ilog2() as usize;
        Ok(Self {
            statement,
            layout,
            constraint_variables,
            identity: KeyIdentity::fresh(),
        })
    }

    /// Returns the checked statement represented by this key.
    #[inline]
    pub const fn statement(&self) -> &Statement<W> {
        // Owning the statement prevents a compiled layout from being paired with another.
        &self.statement
    }

    /// Returns the exact algebraic soundness term for this reduction.
    ///
    /// The width argument is a lower bound on the base-two logarithm of the challenge-field order.
    ///
    /// Every batching and sumcheck challenge must be sampled from that same extension field.
    ///
    /// The result excludes the error of whatever produced the two statement points.
    ///
    /// It also excludes binding and authentication of the returned trace opening.
    #[must_use]
    pub fn security_term(&self, field_bits: NonZeroUsize) -> SecurityTerm {
        // Both batching axes precede both quadratic sumchecks.
        self.security_model(field_bits).combined_term()
    }

    /// Returns separately labelled terms for security-report diagnostics.
    #[must_use]
    pub fn security_components(&self, field_bits: NonZeroUsize) -> Vec<SecurityTerm> {
        // Components must be unioned before composition with another protocol.
        self.security_model(field_bits).components()
    }

    /// Proves that all supplied operand claims read the committed word trace.
    ///
    /// The caller must absorb the trace commitment into the challenger before calling this.
    ///
    /// Skipping that lets a prover choose the trace after seeing the sampled challenges.
    ///
    /// The returned evaluation stays unauthenticated until the caller opens it.
    ///
    /// # Errors
    ///
    /// Returns an error for a misshaped statement point or segment, or a vanishing wiring.
    pub fn prove<F, EF, Challenger>(
        &self,
        values: &PackedWitness<W>,
        claim: &ShiftClaim<EF>,
        challenger: &mut Challenger,
    ) -> Result<(ShiftReductionProof<F, EF>, ShiftOpeningClaim<EF>), ShiftReductionError>
    where
        W: PackedWord,
        F: TranscriptField,
        EF: ExtensionField<F> + Algebra<Gf2>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let (proof, opening, _) = self.prove_with_hints(values, claim, challenger)?;
        Ok((proof, opening))
    }

    /// [`Self::prove`], also returning the values a verifier may defer.
    ///
    /// # Errors
    ///
    /// As [`Self::prove`].
    #[allow(
        clippy::type_complexity,
        reason = "an alias would need bounds it cannot carry"
    )]
    pub fn prove_with_hints<F, EF, Challenger>(
        &self,
        values: &PackedWitness<W>,
        claim: &ShiftClaim<EF>,
        challenger: &mut Challenger,
    ) -> Result<
        (
            ShiftReductionProof<F, EF>,
            ShiftOpeningClaim<EF>,
            ShiftHints<EF>,
        ),
        ShiftReductionError,
    >
    where
        W: PackedWord,
        F: TranscriptField,
        EF: ExtensionField<F> + Algebra<Gf2>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        self.prove_by(values, claim, None, false, false, challenger)
    }

    /// [`Self::prove_with_hints`], the committed words' work on `device` when there is one:
    /// the bit phase's scatter and the whole word phase.
    #[allow(
        clippy::type_complexity,
        reason = "an alias would need bounds it cannot carry"
    )]
    pub(crate) fn prove_by<F, EF, Challenger>(
        &self,
        values: &PackedWitness<W>,
        claim: &ShiftClaim<EF>,
        device: Option<&dyn WordDevice>,
        resident: bool,
        keep: bool,
        challenger: &mut Challenger,
    ) -> Result<
        (
            ShiftReductionProof<F, EF>,
            ShiftOpeningClaim<EF>,
            ShiftHints<EF>,
        ),
        ShiftReductionError,
    >
    where
        W: PackedWord,
        F: TranscriptField,
        EF: ExtensionField<F> + Algebra<Gf2>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Reject shape mismatches before the transcript absorbs any statement value.
        self.validate_claim(claim)?;
        values.check_shape(&self.statement).map_err(|error| {
            ShiftReductionError::SegmentLength {
                segment: error.segment,
                expected: error.expected,
                actual: error.actual,
            }
        })?;

        // Batch relation families first and operand positions second.
        let public_words = values
            .public()
            .iter()
            .copied()
            .map(W::unpack)
            .collect::<Vec<_>>();
        let shape = self.transcript_shape();
        let mut transcript =
            ShiftProverTranscript::<_, F, EF>::new(challenger, shape, claim, &public_words);
        let batch = transcript.batching();
        let prepared = info_span!("shift weights").in_scope(|| PreparedWeights::new(claim, batch));
        let public = public_contribution(
            self.layout.public(),
            &public_words,
            0,
            &prepared,
            claim.bit_point(),
        );
        let witness_claim = prepared.claim(claim) - public;
        let mut device = device.map(|device| {
            info_span!("shift upload").in_scope(|| {
                let words;
                let witness = if resident && keep {
                    Words::Held
                } else if resident {
                    Words::Witness
                } else {
                    words = values
                        .witness()
                        .par_iter()
                        .map(|&packed| W::unpack(packed).to_u64())
                        .collect::<Vec<_>>();
                    Words::Host(&words)
                };
                let (operation, operand) = prepared.batching();
                device.shift(
                    witness,
                    values.witness_len(),
                    W::BITS as usize,
                    self.identity.0,
                    &|| self.device_layout(false),
                    &slice_to_device(operation),
                    &slice_to_device(operand),
                    &slice_to_device(claim.constraint_point()),
                )
            })
        });

        // Phase 1 binds the within-word bit without expanding the committed words.
        let bit_rounds = W::BITS.ilog2() as usize;
        let span = info_span!("shift bit phase").entered();
        let mut bit = match &device {
            Some(device) => {
                let rows = device
                    .bit_rows()
                    .into_iter()
                    .map(from_device)
                    .collect::<Vec<EF>>();
                bit_pairs(&self.layout, &rows, claim.bit_point())
            }
            None => bit_prover(&self.layout, values, &prepared, claim.bit_point()),
        };
        let (bit_sumcheck, bit_point) = transcript.bit_sumcheck(|challenger| {
            bit.prove::<F, _>(challenger, bit_rounds, 2, 0, witness_claim)
        });
        drop(span);
        let intermediate_claim = bit.terminal_sum();
        let mut shifts = bit.right_constants();
        shifts.truncate(self.layout.witness().shift_sequences().len());
        // The word phase can dominate memory, so release both fixed-width tables first.
        drop(bit);

        // Phase 2 binds the committed word index and exposes one trace opening.
        let word_rounds = self.word_variables();
        let (word_sumcheck, word_point, [witness_evaluation, wiring_evaluation]) =
            match device.as_deref_mut() {
                Some(shift) => {
                    info_span!("shift word tables").in_scope(|| {
                        let weights = Point::new(bit_point.as_slice()).equality_weights_msb();
                        let bytes = byte_tables(&weights)
                            .iter()
                            .flat_map(|table| slice_to_device(table))
                            .collect::<Vec<_>>();
                        shift.word_phase(
                            &bytes,
                            &slice_to_device(&shifts),
                            word_rounds,
                            to_device(EF::interpolation_node(2))
                                .expect("only the device's field reaches the device"),
                        );
                    });
                    let mut word = DeviceWordPhase::<EF> {
                        shift: &mut *shift,
                        _field: PhantomData,
                    };
                    let (proof, point) = info_span!("shift word sumcheck").in_scope(|| {
                        transcript.word_sumcheck(|challenger| {
                            word.prove::<F, _>(challenger, word_rounds, 2, 0, intermediate_claim)
                        })
                    });
                    (proof, point, shift.terminal().map(from_device))
                }
                None => {
                    let mut word = info_span!("shift word tables").in_scope(|| {
                        word_prover(
                            &self.layout,
                            values,
                            &prepared,
                            bit_point.as_slice(),
                            &shifts,
                        )
                    });
                    let (proof, point) = info_span!("shift word sumcheck").in_scope(|| {
                        transcript.word_sumcheck(|challenger| {
                            word.prove::<F, _>(challenger, word_rounds, 2, 0, intermediate_claim)
                        })
                    });
                    let constants = word.first_constants();
                    debug_assert_eq!(word.terminal_sum(), constants[0] * constants[1]);
                    (proof, point, constants)
                }
            };
        drop(device);

        // Boolean PCS layout places the within-word coordinates after the word coordinates.
        let opening = ShiftOpeningClaim::new(
            [word_point.as_slice(), bit_point.as_slice()].concat(),
            witness_evaluation,
        );
        transcript.trace_evaluation(witness_evaluation);
        transcript.finish();

        // A vanishing wiring factor leaves the opening value unpinned by the closing equation.
        if wiring_evaluation.is_zero() {
            return Err(ShiftReductionError::DegenerateWiring);
        }
        Ok((
            ShiftReductionProof {
                bit_sumcheck,
                word_sumcheck,
                witness_evaluation,
            },
            opening,
            ShiftHints {
                public,
                wiring: wiring_evaluation,
            },
        ))
    }

    /// Verifies a shift reduction and returns its unauthenticated trace opening.
    ///
    /// The caller must absorb the trace commitment into the challenger before calling this.
    ///
    /// The caller must then discharge the returned claim through the commitment scheme.
    ///
    /// The reduction alone proves nothing about committed data.
    ///
    /// # Errors
    ///
    /// Returns an error for a malformed proof, a vanishing wiring, or a failed closing check.
    pub fn verify<F, EF, Challenger>(
        &self,
        public: &[W],
        claim: &ShiftClaim<EF>,
        proof: &ShiftReductionProof<F, EF>,
        challenger: &mut Challenger,
    ) -> Result<ShiftOpeningClaim<EF>, ShiftReductionError>
    where
        F: TranscriptField,
        EF: ExtensionField<F> + Algebra<Gf2>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        self.verify_with(public, claim, proof, challenger, &mut |deferred, query| {
            self.deferred(deferred, public, public.len(), query)
        })
    }

    /// [`Self::verify`] with the public words' share and the wiring factor taken from
    /// `deferred` instead of evaluated from the compiled relation graph, so a caller can
    /// evaluate them its own way or defer them: see [`Deferred`].
    ///
    /// # Errors
    ///
    /// As [`Self::verify`].
    pub fn verify_with<F, EF, Challenger>(
        &self,
        public: &[W],
        claim: &ShiftClaim<EF>,
        proof: &ShiftReductionProof<F, EF>,
        challenger: &mut Challenger,
        deferred: &mut dyn FnMut(Deferred, &WiringQuery<EF>) -> EF,
    ) -> Result<ShiftOpeningClaim<EF>, ShiftReductionError>
    where
        F: TranscriptField,
        EF: ExtensionField<F> + Algebra<Gf2>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Statement dimensions are checked before any transcript replay.
        self.validate_claim(claim)?;
        if public.len() != self.statement.public_len() {
            return Err(ShiftReductionError::SegmentLength {
                segment: Segment::Public,
                expected: self.statement.public_len(),
                actual: public.len(),
            });
        }

        // Reconstruct the exact batched claim the prover started from.
        let shape = self.transcript_shape();
        let mut transcript =
            ShiftVerifierTranscript::<_, F, EF>::new(challenger, shape, claim, public);
        let batch = transcript.batching();
        let query = WiringQuery {
            constraint_point: claim.constraint_point().to_vec(),
            operation: batch.operation.clone(),
            operand: batch.operand.clone(),
            claim_bit_point: claim.bit_point().to_vec(),
            bit_point: Vec::new(),
            word_point: Vec::new(),
        };
        let prepared = PreparedWeights::sparse(claim, batch);
        let public = deferred(Deferred::Inputs, &query) + deferred(Deferred::Constants, &query);
        let expected_claim = prepared.claim(claim) - public;
        if proof.bit_sumcheck.claimed_sum != expected_claim {
            transcript.abort();
            return Err(ShiftReductionError::EnteringClaim);
        }

        // The two sumchecks meet at the evaluation left by the bit-index rounds.
        let bit_result = transcript.bit_sumcheck(|challenger| {
            proof
                .bit_sumcheck
                .verify(challenger, W::BITS.ilog2() as usize, 2, 0)
        });
        let (bit_point, intermediate_claim) = match bit_result {
            Ok(result) => result,
            Err(error) => {
                transcript.abort();
                return Err(error.into());
            }
        };
        if proof.word_sumcheck.claimed_sum != intermediate_claim {
            transcript.abort();
            return Err(ShiftReductionError::IntermediateClaim);
        }
        let word_result = transcript.word_sumcheck(|challenger| {
            proof
                .word_sumcheck
                .verify(challenger, self.word_variables(), 2, 0)
        });
        let (word_point, final_claim) = match word_result {
            Ok(result) => result,
            Err(error) => {
                transcript.abort();
                return Err(error.into());
            }
        };

        // The wiring factor is public: the caller evaluates it from the compiled relation graph.
        let query = WiringQuery {
            bit_point: bit_point.as_slice().to_vec(),
            word_point: word_point.as_slice().to_vec(),
            ..query
        };
        drop(prepared);
        let wiring = deferred(Deferred::Wiring, &query);
        transcript.trace_evaluation(proof.witness_evaluation);
        transcript.finish();

        // Without this rejection the closing equation holds for every claimed evaluation.
        if wiring.is_zero() {
            return Err(ShiftReductionError::DegenerateWiring);
        }
        if final_claim != proof.witness_evaluation * wiring {
            return Err(ShiftReductionError::FinalClaim);
        }

        // The caller authenticates this value against the committed Boolean trace.
        Ok(ShiftOpeningClaim::new(
            [word_point.as_slice(), bit_point.as_slice()].concat(),
            proof.witness_evaluation,
        ))
    }

    /// The deferred value `deferred` at `query`, for the public words `public` of which those
    /// from `inputs` on are the key's.
    pub fn deferred<F: Field + Algebra<Gf2>>(
        &self,
        deferred: Deferred,
        public: &[W],
        inputs: usize,
        query: &WiringQuery<F>,
    ) -> F {
        let inputs = inputs.min(public.len());
        let (words, first) = match deferred {
            Deferred::Wiring => return self.wiring(query),
            Deferred::Inputs => (&public[..inputs], 0),
            Deferred::Constants => (&public[inputs..], inputs),
        };
        let prepared = PreparedWeights::sparse_from_parts(
            &query.constraint_point,
            query.operation.clone(),
            query.operand.clone(),
        );
        public_contribution(
            self.layout.public(),
            words,
            first,
            &prepared,
            &query.claim_bit_point,
        )
    }

    /// Per public word below `inputs`, the weight of each of its bits in the share
    /// [`Deferred::Inputs`] names at `query`: that share is the sum of the weights of the set
    /// bits, so a caller can evaluate it from the words' bits.
    pub fn input_bit_weights<F: Field + Algebra<Gf2>>(
        &self,
        inputs: usize,
        query: &WiringQuery<F>,
    ) -> Vec<Vec<F>> {
        let prepared = PreparedWeights::sparse_from_parts(
            &query.constraint_point,
            query.operation.clone(),
            query.operand.clone(),
        );
        public_bit_weights(
            self.layout.public(),
            inputs,
            &prepared,
            &query.claim_bit_point,
        )
    }

    /// The wiring factor at `query`, evaluated from the compiled relation graph.
    pub fn wiring<F: Field + Algebra<Gf2>>(&self, query: &WiringQuery<F>) -> F {
        let prepared = PreparedWeights::from_parts(
            &query.constraint_point,
            query.operation.clone(),
            query.operand.clone(),
        );
        let shifts = shift_evaluations(&self.layout, &query.claim_bit_point, &query.bit_point);
        wiring_evaluation(&self.layout, &prepared, &shifts, &query.word_point)
    }

    /// Checks the two public point dimensions.
    fn validate_claim<F>(&self, claim: &ShiftClaim<F>) -> Result<(), ShiftReductionError> {
        // The constraint axis spans the widest zero-padded family.
        if claim.constraint_point().len() != self.constraint_variables {
            return Err(ShiftReductionError::PointLength {
                axis: "constraint",
                expected: self.constraint_variables,
                actual: claim.constraint_point().len(),
            });
        }

        // The bit axis spans every bit of one selected word width.
        let bit_variables = W::BITS.ilog2() as usize;
        if claim.bit_point().len() != bit_variables {
            return Err(ShiftReductionError::PointLength {
                axis: "bit",
                expected: bit_variables,
                actual: claim.bit_point().len(),
            });
        }
        Ok(())
    }

    /// Returns dimensions bound into the batching transcript.
    const fn transcript_shape(&self) -> TranscriptShape {
        // Every count is verifier-derived from the owned system.
        TranscriptShape::new(
            self.constraint_variables,
            W::BITS.ilog2() as usize,
            self.statement.public_len(),
            self.statement.witness_len(),
        )
    }

    /// Returns the number of leading word coordinates in the returned opening point.
    ///
    /// The committed trace must be padded with zero words to exactly that many word slots.
    ///
    /// The remaining coordinates of the point are the within-word bit coordinates.
    #[must_use]
    pub fn word_variables(&self) -> usize {
        // An empty segment is represented by one zero padding slot.
        self.statement
            .witness_len()
            .max(1)
            .next_power_of_two()
            .ilog2() as usize
    }

    /// Returns the number of variables spanning the widest padded relation family.
    #[must_use]
    pub const fn constraint_variables(&self) -> usize {
        // Every family is zero-padded to this shared constraint cube.
        self.constraint_variables
    }

    /// Builds the numeric soundness model from the executed schedule.
    pub(crate) fn security_model(&self, field_bits: NonZeroUsize) -> WordShiftSecurityModel {
        // Two two-variable batching axes and both quadratic round counts are exact.
        WordShiftSecurityModel::new(
            field_bits.get(),
            OPERATION_VARIABLES + OPERAND_VARIABLES,
            W::BITS.ilog2() as usize + self.word_variables(),
        )
        .expect("a nonzero field width gives a valid shift-reduction model")
    }
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;
    use alloc::vec;

    use p3_binary_field::{BinaryChallenger, BinaryField64, BinaryField128, TowerLevel};
    use p3_challenger::HashChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use p3_multilinear_util::point::Point;
    use p3_word::{
        AndConstraint, ConstraintSystem, IntegerMulConstraint, Operand, Shift, ShiftKind,
        ShiftedValue, ValueIndex, Word32, Word64, ZeroConstraint,
    };
    use proptest::prelude::*;

    use super::*;
    use crate::ShiftLayout;
    use crate::shift::wiring::evaluate_claim;

    type F = BinaryField128;
    type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;
    type BaseChallenger = BinaryChallenger<BinaryField64, HashChallenger<u8, Keccak256Hash, 32>>;

    fn challenger() -> Challenger {
        // Prover and verifier begin from the same empty transcript.
        Challenger::from_hasher(Vec::new(), Keccak256Hash)
    }

    fn base_challenger() -> BaseChallenger {
        // This challenger exercises extension-field claims over a narrower transcript field.
        BaseChallenger::from_hasher(Vec::new(), Keccak256Hash)
    }

    fn fixture(
        public_value: u64,
        witness_values: [u64; 3],
    ) -> (
        ShiftReductionKey<Word64>,
        PackedWitness<Word64>,
        Vec<Word64>,
        ShiftClaim<F>,
    ) {
        // The fixture reaches full-width, lane-local, arithmetic, and composed shifts.
        let public_index = ValueIndex::public(0).unwrap();
        let witness = [
            ValueIndex::witness(0).unwrap(),
            ValueIndex::witness(1).unwrap(),
            ValueIndex::witness(2).unwrap(),
        ];
        let rotate = Shift::new(ShiftKind::RotateRight, 13).unwrap();
        let arithmetic = Shift::new(ShiftKind::ArithmeticRight, 7).unwrap();
        let lane = Shift::new(ShiftKind::Lane32RotateRight, 5).unwrap();
        let inner = Shift::new(ShiftKind::LogicalLeft, 9).unwrap();
        let outer = Shift::new(ShiftKind::RotateRight, 17).unwrap();

        let left = Operand::new(vec![
            ShiftedValue::single(witness[0], rotate),
            ShiftedValue::single(public_index, lane),
        ]);
        let right = Operand::new(vec![
            ShiftedValue::single(witness[1], arithmetic),
            ShiftedValue::pair(witness[2], inner, outer).unwrap(),
        ]);
        let output = Operand::new(vec![
            ShiftedValue::plain(witness[0]),
            ShiftedValue::plain(witness[0]),
            ShiftedValue::plain(witness[2]),
        ]);
        let system = ConstraintSystem::new(
            1,
            3,
            vec![ZeroConstraint::new(Operand::new(vec![]))],
            vec![AndConstraint::new(
                left.clone(),
                right.clone(),
                output.clone(),
            )],
            vec![IntegerMulConstraint::new(left.clone(), right, output, left)],
        )
        .unwrap();
        let key = ShiftReductionKey::new(system).unwrap();
        let public = vec![Word64::new(public_value)];
        let witness = witness_values.map(Word64::new).to_vec();
        let packed = PackedWitness::new(
            key.statement()
                .as_system()
                .expect("the fixture is a flat statement"),
            &public,
            &witness,
        )
        .unwrap();
        let constraint_point = vec![];
        let bit_point = (1..=6).map(|bit| F::from_repr(1_u128 << bit)).collect();
        let claim = evaluate_claim(
            key.statement()
                .as_system()
                .expect("the fixture is a flat statement"),
            &packed,
            constraint_point,
            bit_point,
        );
        (key, packed, public, claim)
    }

    #[test]
    fn reduction_round_trips_and_returns_the_direct_trace_evaluation() {
        // Fixture state: one public word and three committed words under five shift spellings.
        let (key, witness, public, claim) = fixture(
            0x0123_4567_89ab_cdef,
            [
                0xfedc_ba98_7654_3210,
                0x8000_0001_7fff_ffff,
                0x0f0f_f0f0_a5a5_5a5a,
            ],
        );
        let mut prover = challenger();
        let (proof, prover_opening) = key
            .prove(&witness, &claim, &mut prover)
            .expect("the honest witness has the checked shape");

        // Verifier replay must recover the identical point and value.
        let mut verifier = challenger();
        let verifier_opening = key
            .verify(&public, &claim, &proof, &mut verifier)
            .expect("the honest reduction closes");
        assert_eq!(prover_opening, verifier_opening);

        // Direct evaluation uses the same word-major, bit-minor Boolean trace layout.
        let (word_point, bit_point) = prover_opening.point().split_at(key.word_variables());
        let word_weights = Point::new(word_point).equality_weights_msb();
        let bit_weights = Point::new(bit_point).equality_weights_msb();
        let expected = witness
            .witness()
            .iter()
            .enumerate()
            .map(|(word, &packed)| {
                word_weights[word] * wiring::evaluate_word(Word64::unpack(packed), &bit_weights)
            })
            .sum::<F>();
        assert_eq!(prover_opening.value(), expected);
    }

    #[test]
    fn tampering_each_sumcheck_or_opening_is_rejected() {
        // Start from one valid proof so every mutation targets one protocol boundary.
        let (key, witness, public, claim) = fixture(3, [5, 7, 11]);
        let mut prover = challenger();
        let (proof, _) = key.prove(&witness, &claim, &mut prover).unwrap();

        // Mutation 1: perturb the first bit-round message.
        //
        // The omitted value at one is reconstructed from the running claim.
        //
        // The round stays self-consistent, so the phase ends elsewhere instead of rejecting.
        let mut bit = proof.clone();
        bit.bit_sumcheck.round_polys[0][0] += F::ONE;
        assert_eq!(
            key.verify(&public, &claim, &bit, &mut challenger()),
            Err(ShiftReductionError::IntermediateClaim)
        );

        // Mutation 2: the word phase claims a different intermediate value.
        let mut bridge = proof.clone();
        bridge.word_sumcheck.claimed_sum += F::ONE;
        assert_eq!(
            key.verify(&public, &claim, &bridge, &mut challenger()),
            Err(ShiftReductionError::IntermediateClaim)
        );

        // Mutation 3: public words are part of the statement bound before batching.
        //
        // Their share is subtracted from the claim the bit phase starts at.
        //
        // The first boundary therefore rejects, before either sumcheck runs.
        let changed_public = [Word64::new(public[0].get() ^ 1)];
        assert_eq!(
            key.verify(&changed_public, &claim, &proof, &mut challenger()),
            Err(ShiftReductionError::EnteringClaim)
        );

        // Mutation 4: the final PCS opening no longer closes the word sumcheck.
        let mut opening = proof;
        opening.witness_evaluation += F::ONE;
        assert_eq!(
            key.verify(&public, &claim, &opening, &mut challenger()),
            Err(ShiftReductionError::FinalClaim)
        );
    }

    #[test]
    fn changing_any_public_operand_claim_is_rejected() {
        // Every semantic role is public input and precedes both batching challenges.
        let (key, witness, public, claim) = fixture(29, [31, 37, 41]);
        let mut prover = challenger();
        let (proof, _) = key.prove(&witness, &claim, &mut prover).unwrap();

        for changed in 0..8 {
            let mut zero = *claim.zero();
            let mut bitwise_and = *claim.bitwise_and();
            let mut integer_mul = *claim.integer_mul();
            match changed {
                0 => zero[0] += F::ONE,
                1..=3 => bitwise_and[changed - 1] += F::ONE,
                4..=7 => integer_mul[changed - 4] += F::ONE,
                _ => unreachable!(),
            }
            let changed_claim = ShiftClaim::new(
                claim.constraint_point().to_vec(),
                claim.bit_point().to_vec(),
                zero,
                bitwise_and,
                integer_mul,
            );

            assert_eq!(
                key.verify(&public, &changed_claim, &proof, &mut challenger()),
                Err(ShiftReductionError::EnteringClaim),
                "claim position {changed} was not bound",
            );
        }
    }

    #[test]
    fn changing_constraint_order_changes_the_verifier_wiring() {
        // Two rows make the constraint coordinate and its compiled reference order observable.
        let first = ValueIndex::witness(0).unwrap();
        let second = ValueIndex::witness(1).unwrap();
        let left = Shift::new(ShiftKind::LogicalLeft, 7).unwrap();
        let rotate = Shift::new(ShiftKind::RotateRight, 11).unwrap();
        let first_relation =
            ZeroConstraint::new(Operand::single(ShiftedValue::single(first, left)));
        let second_relation =
            ZeroConstraint::new(Operand::single(ShiftedValue::single(second, rotate)));
        let system = ConstraintSystem::new(
            0,
            2,
            vec![first_relation.clone(), second_relation.clone()],
            vec![],
            vec![],
        )
        .unwrap();
        let reordered =
            ConstraintSystem::new(0, 2, vec![second_relation, first_relation], vec![], vec![])
                .unwrap();
        let key = ShiftReductionKey::new(system).unwrap();
        let reordered_key = ShiftReductionKey::new(reordered).unwrap();
        let words = [Word64::new(0x0123), Word64::new(0x4567)];
        let witness = PackedWitness::new(
            key.statement()
                .as_system()
                .expect("the fixture is a flat statement"),
            &[],
            &words,
        )
        .unwrap();
        let claim = evaluate_claim(
            key.statement()
                .as_system()
                .expect("the fixture is a flat statement"),
            &witness,
            vec![F::from_repr(1 << 17)],
            (18..24).map(|bit| F::from_repr(1 << bit)).collect(),
        );
        let mut prover = challenger();
        let (proof, _) = key.prove(&witness, &claim, &mut prover).unwrap();

        // The shape and transcript inputs agree, but the verifier-fixed row wiring does not.
        assert_eq!(
            reordered_key.verify(&[], &claim, &proof, &mut challenger()),
            Err(ShiftReductionError::FinalClaim)
        );
    }

    #[test]
    fn challenge_field_can_extend_the_transcript_field() {
        // The reduction must not force commitments and challenges into one field type.
        let (key, witness, public, claim) = fixture(13, [17, 19, 23]);
        let mut prover = base_challenger();
        let (proof, expected) = key.prove(&witness, &claim, &mut prover).unwrap();
        let mut verifier = base_challenger();
        let actual = key.verify(&public, &claim, &proof, &mut verifier).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn malformed_statement_points_are_rejected_before_replay() {
        // Word32 needs five bit coordinates and this statement has no constraint coordinate.
        let system = ConstraintSystem::<Word32>::new(0, 0, vec![], vec![], vec![]).unwrap();
        let key = ShiftReductionKey::new(system).unwrap();
        let claim = ShiftClaim::new(
            vec![F::ZERO],
            vec![F::ZERO; 4],
            [F::ZERO],
            [F::ZERO; 3],
            [F::ZERO; 4],
        );
        let proof = ShiftReductionProof {
            bit_sumcheck: GenericDegreeProof::default(),
            word_sumcheck: GenericDegreeProof::default(),
            witness_evaluation: F::ZERO,
        };

        // Constraint length is checked first and reports both dimensions.
        assert_eq!(
            key.verify(&[], &claim, &proof, &mut challenger()),
            Err(ShiftReductionError::PointLength {
                axis: "constraint",
                expected: 0,
                actual: 1,
            })
        );
    }

    #[test]
    fn a_system_with_no_committed_reference_pins_no_opening() {
        // An empty committed segment makes the public wiring identically zero.
        let system = ConstraintSystem::<Word32>::new(0, 0, vec![], vec![], vec![]).unwrap();
        let key = ShiftReductionKey::new(system).unwrap();
        let witness = PackedWitness::new(
            key.statement()
                .as_system()
                .expect("the fixture is a flat statement"),
            &[],
            &[],
        )
        .unwrap();
        let claim = ShiftClaim::new(
            vec![],
            (24..29).map(|bit| F::from_repr(1 << bit)).collect(),
            [F::ZERO],
            [F::ZERO; 3],
            [F::ZERO; 4],
        );

        // Proving stops rather than handing back an evaluation nothing constrains.
        assert_eq!(
            key.prove::<F, F, _>(&witness, &claim, &mut challenger())
                .expect_err("a vanishing wiring factor stops the prover"),
            ShiftReductionError::DegenerateWiring
        );

        // The closing equation reads zero equals anything times zero, so it accepts every value.
        // Five all-zero quadratic rounds and no word round replay without complaint.
        for value in [F::ZERO, F::ONE, F::from_repr(1 << 33)] {
            let forged = ShiftReductionProof::<F, F> {
                bit_sumcheck: GenericDegreeProof {
                    claimed_sum: F::ZERO,
                    round_polys: vec![vec![F::ZERO; 2]; 5],
                    pow_witnesses: vec![],
                },
                word_sumcheck: GenericDegreeProof {
                    claimed_sum: F::ZERO,
                    round_polys: vec![],
                    pow_witnesses: vec![],
                },
                witness_evaluation: value,
            };
            assert_eq!(
                key.verify(&[], &claim, &forged, &mut challenger()),
                Err(ShiftReductionError::DegenerateWiring)
            );
        }
    }

    #[test]
    fn word32_shifts_close_the_five_round_bit_phase() {
        // Fixture state: two committed 32-bit words under all four full-width movements.
        let first = ValueIndex::witness(0).unwrap();
        let second = ValueIndex::witness(1).unwrap();
        let logical_left = Shift::new(ShiftKind::LogicalLeft, 5).unwrap();
        let logical_right = Shift::new(ShiftKind::LogicalRight, 11).unwrap();
        let arithmetic = Shift::new(ShiftKind::ArithmeticRight, 9).unwrap();
        let rotate = Shift::new(ShiftKind::RotateRight, 27).unwrap();
        let left = Operand::new(vec![
            ShiftedValue::single(first, logical_left),
            ShiftedValue::single(second, arithmetic),
        ]);
        let right = Operand::new(vec![
            ShiftedValue::single(second, logical_right),
            ShiftedValue::pair(first, logical_right, rotate).unwrap(),
        ]);
        let output = Operand::single(ShiftedValue::plain(second));
        let system = ConstraintSystem::<Word32>::new(
            0,
            2,
            vec![],
            vec![AndConstraint::new(left, right, output)],
            vec![],
        )
        .unwrap();
        let key = ShiftReductionKey::new(system).unwrap();

        // A set high bit in each word drives the sign-extending movement.
        let words = [Word32::new(0x8765_4321), Word32::new(0xfedc_ba98)];
        let witness = PackedWitness::new(
            key.statement()
                .as_system()
                .expect("the fixture is a flat statement"),
            &[],
            &words,
        )
        .unwrap();
        let claim = evaluate_claim(
            key.statement()
                .as_system()
                .expect("the fixture is a flat statement"),
            &witness,
            vec![],
            (1..6).map(|bit| F::from_repr(1 << bit)).collect(),
        );
        let mut prover = challenger();
        let (proof, expected) = key.prove(&witness, &claim, &mut prover).unwrap();
        let actual = key.verify(&[], &claim, &proof, &mut challenger()).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn lane_local_and_full_width_logical_right_close_end_to_end() {
        // These four movements are the ones the shared 64-bit fixture never reaches.
        let first = ValueIndex::witness(0).unwrap();
        let second = ValueIndex::witness(1).unwrap();
        let lane_arithmetic = Shift::new(ShiftKind::Lane32ArithmeticRight, 9).unwrap();
        let lane_left = Shift::new(ShiftKind::Lane32LogicalLeft, 13).unwrap();
        let lane_right = Shift::new(ShiftKind::Lane32LogicalRight, 21).unwrap();
        let logical_right = Shift::new(ShiftKind::LogicalRight, 35).unwrap();
        let left = Operand::new(vec![
            ShiftedValue::single(first, lane_arithmetic),
            ShiftedValue::single(second, lane_left),
        ]);
        let right = Operand::new(vec![
            ShiftedValue::single(second, lane_right),
            ShiftedValue::single(first, logical_right),
        ]);
        let output = Operand::single(ShiftedValue::plain(first));
        let system = ConstraintSystem::new(
            0,
            2,
            vec![],
            vec![AndConstraint::new(left, right, output)],
            vec![],
        )
        .unwrap();
        let key = ShiftReductionKey::new(system).unwrap();

        // Both 32-bit lanes carry a set sign bit so per-lane fan-out is observable.
        let words = [
            Word64::new(0x8000_0001_8765_4321),
            Word64::new(0x1234_5678_9abc_def0),
        ];
        let witness = PackedWitness::new(
            key.statement()
                .as_system()
                .expect("the fixture is a flat statement"),
            &[],
            &words,
        )
        .unwrap();
        let claim = evaluate_claim(
            key.statement()
                .as_system()
                .expect("the fixture is a flat statement"),
            &witness,
            vec![],
            (7..13).map(|bit| F::from_repr(1 << bit)).collect(),
        );
        let mut prover = challenger();
        let (proof, expected) = key.prove(&witness, &claim, &mut prover).unwrap();
        let actual = key.verify(&[], &claim, &proof, &mut challenger()).unwrap();

        assert_eq!(actual, expected);
    }

    /// The shift device's contract run on the host, from the flattened wiring alone.
    struct HostShiftDevice;

    struct HostShift {
        words: Vec<u64>,
        width: usize,
        layout: ShiftLayout,
        operation: Vec<F>,
        operand: Vec<F>,
        constraint: Vec<F>,
        factors: [Vec<F>; 2],
        two: F,
    }

    impl HostShift {
        /// Every key of word `w`: its sequence and coefficient.
        fn coefficients(&self, w: usize) -> Vec<(usize, F)> {
            let w = w as u32;
            let block = self
                .layout
                .blocks
                .chunks(10)
                .find(|b| b[0] <= w && w < b[0] + b[1] * b[2])
                .expect("a block owns every word");
            let (instance, slot) = ((w - block[0]) / block[1], (w - block[0]) % block[1]);
            let at = (block[3] + slot) as usize;
            let (first, last) = (
                self.layout.slot_keys[2 * at],
                self.layout.slot_keys[2 * at + 1],
            );
            (first..last)
                .map(|k| {
                    let key = &self.layout.keys[4 * k as usize..][..4];
                    let family = key[0] as usize;
                    let offset = block[7 + family] + instance * block[4 + family];
                    let sum = (key[2]..key[3])
                        .map(|r| {
                            self.operand[self.layout.operands[r as usize] as usize]
                                * self.constraint
                                    [(offset + self.layout.constraints[r as usize]) as usize]
                        })
                        .sum::<F>();
                    (key[1] as usize, self.operation[family] * sum)
                })
                .collect()
        }
    }

    impl DeviceShift for HostShift {
        fn bit_rows(&self) -> Vec<u128> {
            let mut rows = vec![F::ZERO; self.layout.shifts * self.width];
            for (w, &word) in self.words.iter().enumerate() {
                for (s, c) in self.coefficients(w) {
                    for b in (0..self.width).filter(|&b| (word >> b) & 1 == 1) {
                        rows[s * self.width + b] += c;
                    }
                }
            }
            slice_to_device(&rows)
        }

        fn word_phase(
            &mut self,
            bytes: &[u128],
            shift_evaluations: &[u128],
            vars: usize,
            two: u128,
        ) {
            let bytes = bytes.iter().map(|&x| F::from_repr(x)).collect::<Vec<_>>();
            let shifts = shift_evaluations
                .iter()
                .map(|&x| F::from_repr(x))
                .collect::<Vec<_>>();
            let mut left = vec![F::ZERO; 1 << vars];
            let mut right = vec![F::ZERO; 1 << vars];
            for (w, &word) in self.words.iter().enumerate() {
                left[w] = (0..self.width / 8)
                    .map(|byte| bytes[byte * 256 + ((word >> (8 * byte)) & 0xff) as usize])
                    .sum();
                right[w] = self
                    .coefficients(w)
                    .into_iter()
                    .map(|(s, c)| c * shifts[s])
                    .sum();
            }
            self.factors = [left, right];
            self.two = F::from_repr(two);
        }

        fn round(&self) -> [u128; 2] {
            let [left, right] = &self.factors;
            let half = left.len() / 2;
            let (mut zero, mut two) = (F::ZERO, F::ZERO);
            for row in 0..half {
                let (l0, l1, r0, r1) = (left[row], left[row + half], right[row], right[row + half]);
                zero += l0 * r0;
                two += (l0 + (l1 - l0) * self.two) * (r0 + (r1 - r0) * self.two);
            }
            [zero.to_repr(), two.to_repr()]
        }

        fn fold(&mut self, challenge: u128) {
            let c = F::from_repr(challenge);
            for factor in &mut self.factors {
                let half = factor.len() / 2;
                for row in 0..half {
                    let (low, high) = (factor[row], factor[row + half]);
                    factor[row] = low + (high - low) * c;
                }
                factor.truncate(half);
            }
        }

        fn terminal(&self) -> [u128; 2] {
            self.factors.each_ref().map(|f| f[0].to_repr())
        }
    }

    impl WordDevice for HostShiftDevice {
        fn power_tree(
            &self,
            _: &[u128],
            _: Words<'_>,
            _: Words<'_>,
            _: usize,
            _: usize,
            _: usize,
        ) -> Box<dyn crate::DeviceTree> {
            unreachable!("the shift reduction builds no tree")
        }

        fn gather_witness(&self, _: &[u32]) -> bool {
            unreachable!("the test gathers no words")
        }

        fn port_trees(
            &self,
            _: usize,
            _: usize,
            _: &[u128],
            _: &[u128],
            _: u128,
            _: u128,
            _: &[u32],
        ) -> Option<Box<dyn crate::DevicePorts>> {
            unreachable!("the shift reduction builds no port trees")
        }

        fn bit_tree(
            &self,
            _: &[Words<'_>],
            _: &[u128],
            _: usize,
            _: usize,
            _: usize,
        ) -> Box<dyn crate::DeviceTree> {
            unreachable!("the shift reduction builds no tree")
        }

        fn bit_values(&self, _: &[Words<'_>], _: usize, _: &[u128]) -> Vec<u128> {
            unreachable!("the shift reduction reads no bit values")
        }

        fn selector(
            &self,
            _: &[Vec<u128>],
            _: &[u128],
            _: Words<'_>,
            _: &[u128],
            _: Words<'_>,
            _: usize,
            _: [u128; 3],
        ) -> Box<dyn crate::DeviceSelector> {
            unreachable!("the shift reduction builds no selector")
        }

        fn relation(
            &self,
            _: &[Words<'_>],
            _: &[Vec<u128>],
            _: usize,
        ) -> Box<dyn crate::DeviceRelation> {
            unreachable!("the shift reduction checks no relation")
        }

        fn shift(
            &self,
            witness: Words<'_>,
            _: usize,
            width: usize,
            _: u64,
            layout: &dyn Fn() -> ShiftLayout,
            operation: &[u128],
            operand: &[u128],
            constraint_point: &[u128],
        ) -> Box<dyn DeviceShift> {
            let point = constraint_point
                .iter()
                .map(|&x| F::from_repr(x))
                .collect::<Vec<_>>();
            let field = |values: &[u128]| values.iter().map(|&x| F::from_repr(x)).collect();
            Box::new(HostShift {
                words: witness.host().to_vec(),
                width,
                layout: layout(),
                operation: field(operation),
                operand: field(operand),
                constraint: Point::new(point.as_slice()).equality_weights_msb(),
                factors: [Vec::new(), Vec::new()],
                two: F::ZERO,
            })
        }
    }

    #[test]
    fn the_device_contract_reaches_the_host_proof() {
        // Several relations of every family put weight on the constraint rows.
        let words = (0..11)
            .map(|i| ValueIndex::witness(i).unwrap())
            .collect::<Vec<_>>();
        let public = ValueIndex::public(0).unwrap();
        let kinds = [
            ShiftKind::RotateRight,
            ShiftKind::LogicalLeft,
            ShiftKind::ArithmeticRight,
            ShiftKind::Lane32RotateRight,
            ShiftKind::LogicalRight,
        ];
        let operand = |i: usize| {
            let shift = Shift::new(kinds[i % kinds.len()], 7 * i % 31 + 1).unwrap();
            Operand::new(vec![
                ShiftedValue::single(words[i % words.len()], shift),
                ShiftedValue::plain(words[(3 * i + 1) % words.len()]),
                ShiftedValue::single(public, Shift::new(ShiftKind::RotateRight, 3).unwrap()),
            ])
        };
        let system = ConstraintSystem::new(
            1,
            words.len(),
            (0..3).map(|i| ZeroConstraint::new(operand(i))).collect(),
            (0..6)
                .map(|i| AndConstraint::new(operand(i + 3), operand(i + 4), operand(i + 5)))
                .collect(),
            (0..2)
                .map(|i| {
                    IntegerMulConstraint::new(
                        operand(i),
                        operand(i + 1),
                        operand(i + 2),
                        operand(i + 6),
                    )
                })
                .collect(),
        )
        .unwrap();
        let key = ShiftReductionKey::new(system).unwrap();
        let public_words = vec![Word64::new(0x0123_4567_89ab_cdef)];
        let witness = (0..words.len() as u64)
            .map(|i| Word64::new(i.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (i << 40)))
            .collect::<Vec<_>>();
        let system = key.statement().as_system().expect("a flat statement");
        let packed = PackedWitness::new(system, &public_words, &witness).unwrap();
        let constraint_point = (0..key.constraint_variables())
            .map(|i| F::from_repr(0x1234_5678_u128 << (3 * i) | 5))
            .collect();
        let bit_point = (1..=6).map(|bit| F::from_repr(3_u128 << bit)).collect();
        let claim = evaluate_claim(system, &packed, constraint_point, bit_point);

        let host = key
            .prove_by::<F, F, _>(&packed, &claim, None, false, false, &mut challenger())
            .unwrap();
        let device = key
            .prove_by::<F, F, _>(&packed, &claim, Some(&HostShiftDevice), false, false, &mut challenger())
            .unwrap();
        assert_eq!(
            postcard::to_allocvec(&host.0).unwrap(),
            postcard::to_allocvec(&device.0).unwrap()
        );
        assert_eq!(host.1, device.1);
    }

    proptest! {
        #[test]
        fn random_words_match_the_scalar_reference(
            public in any::<u64>(),
            witness in prop::array::uniform3(any::<u64>()),
        ) {
            // Random bit patterns exercise sign extension and both 32-bit lanes.
            let (key, values, public_words, claim) = fixture(public, witness);
            let mut prover = challenger();
            let (proof, expected) = key.prove(&values, &claim, &mut prover).unwrap();
            let mut verifier = challenger();
            let actual = key.verify(&public_words, &claim, &proof, &mut verifier).unwrap();

            prop_assert_eq!(actual, expected);
        }
    }
}
