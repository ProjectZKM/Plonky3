//! A bit witness committed narrow, and opened through WHIR over the additive binary domain.
//!
//! ```text
//!     packing    bits  ->  the elements a commitment holds, by reinterpretation
//!     switch     a claim about the bits  ->  a claim about the packing
//!     opening    that claim  ->  discharged by WHIR
//! ```
//!
//! The packing is a bijection between bit strings and elements of the alphabet.
//!
//! One committed byte therefore holds eight trace bits, whatever the alphabet's width.
//!
//! No step of the commit path widens a bit into an element of its own.
//!
//! # Soundness
//!
//! Three errors compose by a union bound.
//!
//! ```text
//!     commitment       the proximity argument's own budget, at one opened point
//!     reduction        one bit ring switch for every claim, its rounds run once
//!     claim batching   the claims folded under powers of lambda
//! ```
//!
//! All come back labelled, so a report says which one is short.
//!
//! The batch runs before the proximity argument names one codeword.
//!
//! So it is charged over every candidate the commitment still leaves open.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{BitCoordinates, PackedGf2, TowerLevel, Underlier};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_security::multilinear::{bit_ring_switch_claim_batching_term, bit_ring_switch_tensors_term};
use p3_security::{ErrorBits, SecurityTerm};
use p3_sumcheck::layout::{Layout, SuffixProver, Table};
use p3_sumcheck::ring_switch::bits::{
    BitPacking, BitPackingView, BitRingSwitch, BitRingSwitchClaims, BitRingSwitchClaimsProof,
};
use p3_sumcheck::{
    OpeningBatch, OpeningProtocol, PrescribedOpeningSecurity, PrescribedPointPcs, TableShape,
    TableSpec,
};
use p3_whir::{WhirDomain, WhirProver, WhirProverData, pair_batching_error};

use crate::boolean::{
    BitOpening, BitReadings, BooleanBackend, BooleanMultilinearPcs, PairReadings, ReadingsCheck,
};
use crate::boolean_trace::BooleanTraceCommitment;
use crate::fold::BitChallengeField;
use crate::packing::{Coordinates, PackedStack, PackedWords, hypercube_variables};
use crate::whir::error::BooleanWhirError;
use crate::whir::proof::{BooleanWhirOpening, BooleanWhirProof};
use crate::whir::shape::ProofShape;

/// Label of the error a pair opening's batching draw charges.
pub const PAIR_BATCHING_LABEL: &str = "pair-batching";

/// The binding mode the committed layout uses.
///
/// One mode is fixed rather than chosen, so a commitment and its replay never disagree.
type Binding<F, EF> = SuffixProver<F, EF>;

/// The proximity argument every packed claim is discharged against.
///
/// It commits the packing level `F` and draws its challenges from `EF`.
pub type BooleanWhirProver<F, EF, Dft, MT, Challenger> =
    WhirProver<EF, F, Dft, MT, Challenger, Binding<F, EF>>;

/// Prover-side data retained between committing and opening.
pub type BooleanWhirData<F, EF, MT> = WhirProverData<F, EF, MT, Binding<F, EF>>;

/// A batched trace of Boolean columns, discharged through the additive-domain proximity argument.
pub type BooleanWhirTracePcs<F, EF, Dft, MT, Challenger> =
    BooleanTraceCommitment<EF, BooleanWhirPcs<F, EF, Dft, MT, Challenger>>;

/// A commitment to a function from the hypercube to `{0, 1}`, opened through WHIR.
///
/// The committed object is a bit witness, packed into the tower level `F`.
///
/// An opening answers for its multilinear extension at a point of the challenge field `EF`.
///
/// ```text
///     F = EF = GF(2^128)                 128 bits per committed element
///     F = GF(2^64),  EF = GF(2^192)      64 bits per committed element, wider challenges
/// ```
pub struct BooleanWhirPcs<F: Field, EF: ExtensionField<F>, Dft, MT, Challenger> {
    /// The proximity argument the packed multilinear is discharged against.
    inner: BooleanWhirProver<F, EF, Dft, MT, Challenger>,
    /// Variables the bit witness has, which is the packing's plus the absorbed ones.
    num_variables: usize,
}

impl<F, EF, Dft, MT, Challenger> BooleanWhirPcs<F, EF, Dft, MT, Challenger>
where
    F: Field + TranscriptField + TowerLevel + Coordinates + Ord,
    EF: BitCoordinates + ExtensionField<F>,
    Dft: WhirDomain<F, EF>,
    MT: Mmcs<F>,
{
    /// Wrap a proximity argument as a commitment to a bit witness of this many variables.
    ///
    /// # Errors
    ///
    /// Returns an error unless one element can absorb the witness's low coordinates.
    ///
    /// Returns an error unless the schedule commits exactly the elements the packing holds.
    pub fn new(
        inner: BooleanWhirProver<F, EF, Dft, MT, Challenger>,
        num_variables: usize,
    ) -> Result<Self, BooleanWhirError> {
        let packed = Self::packed_variables(num_variables)?;
        if inner.num_variables() != packed {
            return Err(BooleanWhirError::ConfigArity {
                expected: packed,
                actual: inner.num_variables(),
            });
        }
        Ok(Self {
            inner,
            num_variables,
        })
    }

    /// Variables the committed bit witness has, so `2^n` bits in all.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Bytes the commitment holds, which is one per eight bits of the witness.
    #[must_use]
    pub const fn committed_bytes(&self) -> usize {
        1 << (self.num_variables - 3)
    }

    /// Variables the packing keeps, the witness's less the ones one element absorbs.
    fn packed_variables(num_variables: usize) -> Result<usize, BooleanWhirError> {
        let absorbed = BitRingSwitch::<F, EF>::ABSORBED;
        num_variables
            .checked_sub(absorbed)
            .ok_or(BooleanWhirError::WitnessTooNarrow {
                needed: absorbed,
                actual: num_variables,
            })
    }

    /// The opening schedule this many surviving claims are discharged through.
    ///
    /// One table of one column, opened directly at one point per claim.
    /// A batch of claims leaves one surviving claim, whatever its size.
    fn protocol(&self, num_claims: usize) -> OpeningProtocol {
        OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(self.inner.num_variables(), 1),
            (0..num_claims)
                .map(|_| OpeningBatch::new(vec![0], Vec::new()))
                .collect(),
        )])
    }

    /// Every opening's reduction, gathered into the one batch that reduces them all.
    fn claims(openings: &[BitOpening<EF>]) -> Result<BitRingSwitchClaims<F, EF>, BooleanWhirError> {
        let reductions = openings
            .iter()
            .map(Self::reduction)
            .collect::<Result<Vec<_>, _>>()?;
        BitRingSwitchClaims::new(reductions).map_err(BooleanWhirError::Reduction)
    }

    /// The reduction answering every reading one opening asks for.
    fn reduction(opening: &BitOpening<EF>) -> Result<BitRingSwitch<F, EF>, BooleanWhirError> {
        if opening.next {
            BitRingSwitch::with_successor(&opening.point, opening.row_variables)
        } else {
            BitRingSwitch::new(&opening.point)
        }
        .map_err(BooleanWhirError::Reduction)
    }

    /// Each point as an opening asking for the current reading alone.
    fn current_openings(points: &[Point<EF>]) -> Vec<BitOpening<EF>> {
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

    /// Check that every opening names the witness's variables and asks for a reading it has.
    fn check_openings(&self, openings: &[BitOpening<EF>]) -> Result<(), BooleanWhirError> {
        if openings.is_empty() {
            return Err(BooleanWhirError::NoPoints);
        }
        for (index, opening) in openings.iter().enumerate() {
            if opening.point.num_variables() != self.num_variables {
                return Err(BooleanWhirError::PointArity {
                    expected: self.num_variables,
                    actual: opening.point.num_variables(),
                });
            }
            if !opening.current && !opening.next {
                return Err(BooleanWhirError::NoReading { index });
            }
            if opening.row_variables > self.num_variables {
                return Err(BooleanWhirError::RowVariables {
                    index,
                    row_variables: opening.row_variables,
                    num_variables: self.num_variables,
                });
            }
        }
        Ok(())
    }

    /// The packed multilinear the commitment holds, borrowed from the retained table.
    ///
    /// # Panics
    ///
    /// Never for a table this scheme committed.
    ///
    /// Its alphabet is byte aligned and its height is a power of two.
    fn packing(prover_data: &BooleanWhirData<F, EF, MT>) -> BitPackingView<'_, F> {
        BitPacking::from_packed(prover_data.table(0).poly(0))
            .expect("a committed table is a hypercube over a byte-aligned level")
    }
}

impl<F, EF, Dft, MT, Challenger> BooleanWhirPcs<F, EF, Dft, MT, Challenger>
where
    F: Field + TranscriptField + TowerLevel + Coordinates + Ord + Send + Sync,
    EF: BitChallengeField<F>,
    Dft: WhirDomain<F, EF>,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<MT::Commitment>,
{
    /// Every labelled algebraic error one opening of this many claims charges.
    ///
    /// ```text
    ///     commitment       the proximity argument's own budget, at the one surviving point
    ///     reduction        one batched bit ring switch, batching its elements under one draw
    ///     claim batching   the claims folded under powers of lambda
    /// ```
    ///
    /// The terms are independent draws, so they compose by a union bound.
    ///
    /// The reduction is charged over every candidate the commitment leaves open.
    ///
    /// Its challenges are drawn before the proximity argument names one of them.
    ///
    /// The count itself is forwarded untouched, because this is a link and not the end.
    ///
    /// A caller stacking its own draw faces the same list the proximity argument left.
    ///
    /// So it charges that draw over the same count.
    ///
    /// Nothing here covers hash or transcript collisions, which the caller supplies.
    ///
    /// # Returns
    ///
    /// Nothing when the proximity argument declines to price the schedule.
    #[must_use]
    pub fn readings_security(
        &self,
        num_claims: usize,
        successor_tensors: bool,
    ) -> Option<PrescribedOpeningSecurity> {
        // Every claim of a batch survives as one claim, at one point.
        let protocol = self.protocol(num_claims.min(1));
        let mut security = self.inner.prescribed_security(&protocol)?;
        // The batch runs before one candidate is named, so it pays for all of them.
        //
        // The charge leaves the count alone, so the caller above still sees the same list.
        //
        // It then charges its own draws over that list.
        for term in Self::reduction_terms(self.inner.num_variables(), num_claims, successor_tensors)
        {
            security.charge_reduction(term);
        }
        Some(security)
    }

    /// The shape of every proof an opening of this many claims can produce.
    ///
    /// It covers the reductions as well as the opening, so a ceiling graded against it binds both.
    #[must_use]
    pub fn proof_shape(&self, num_claims: usize, successor_tensors: bool) -> ProofShape {
        ProofShape::of_bit_readings(&self.inner, num_claims, successor_tensors)
    }

    /// Commit to packed elements as the one column of the one table the layout stacks.
    #[allow(clippy::type_complexity)]
    fn commit_column(
        &self,
        elements: Vec<F>,
        challenger: &mut Challenger,
    ) -> Result<(MT::Commitment, BooleanWhirData<F, EF, MT>), BooleanWhirError> {
        let actual = hypercube_variables(elements.len()).map_err(BooleanWhirError::Packing)?;
        if actual != self.inner.num_variables() {
            return Err(BooleanWhirError::WitnessArity {
                expected: self.inner.num_variables(),
                actual,
            });
        }

        let len = elements.len();
        let table = Table::new(RowMajorMatrix::new(elements, len));
        let folding = self.inner.round_folding_factor(0);
        let witness = Binding::<F, EF>::new_witness(vec![table], folding);
        p3_commit::MultilinearPcs::<EF, Challenger>::commit(&self.inner, witness, challenger)
            .map_err(BooleanWhirError::Commit)
    }

    /// Run one batched ring switch over every opening, leaving one claim about the packing.
    ///
    /// # Returns
    ///
    /// The readings, the reduction's proof, and the surviving point.
    #[allow(clippy::type_complexity)]
    fn switch(
        &self,
        prover_data: &BooleanWhirData<F, EF, MT>,
        openings: &[BitOpening<EF>],
        challenger: &mut Challenger,
    ) -> Result<
        (
            Vec<BitReadings<EF>>,
            BitRingSwitchClaimsProof<F, EF>,
            Point<EF>,
        ),
        BooleanWhirError,
    > {
        self.check_openings(openings)?;
        // Every reduction is set up before any runs, so a refused one leaves the transcript alone.
        let claims = Self::claims(openings)?;
        let packing = Self::packing(prover_data);

        // One batch for every opening, leaving one claim about the packing.
        let (reduction, surviving_point, _) =
            tracing::info_span!("bit ring switch").in_scope(|| {
                claims
                    .prove::<<EF as BitChallengeField<F>>::SumcheckRepr, _, _>(&packing, challenger)
            });

        // The elements each claim sends already hold its readings.
        let readings = openings
            .iter()
            .zip(claims.reductions())
            .zip(&reduction.claims)
            .map(|((opening, switch), elements)| {
                let current = opening
                    .current
                    .then(|| switch.incoming_claim(&elements.tensor));
                let next = opening
                    .next
                    .then(|| switch.successor_claim(&elements.tensor, elements.successor.as_ref()))
                    .transpose()
                    .map_err(BooleanWhirError::Reduction)?;
                Ok(BitReadings { current, next })
            })
            .collect::<Result<Vec<_>, BooleanWhirError>>()?;
        Ok((readings, reduction, surviving_point))
    }

    /// Replay one batched ring switch against the readings a proof claims.
    ///
    /// # Returns
    ///
    /// The surviving point, and the value the packing must hold there.
    fn check_switch(
        &self,
        openings: &[BitOpening<EF>],
        readings: &[BitReadings<EF>],
        proof: &BooleanWhirProof<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> Result<(Point<EF>, EF), BooleanWhirError> {
        self.check_openings(openings)?;
        if readings.len() != openings.len() || proof.reduction.claims.len() != openings.len() {
            return Err(BooleanWhirError::ClaimCount {
                expected: openings.len(),
                values: readings.len(),
                reductions: proof.reduction.claims.len(),
            });
        }
        // The openings fix which readings are checked, so a missing one is never skipped.
        if let Some(index) = openings
            .iter()
            .zip(readings)
            .position(|(opening, reading)| {
                reading.current.is_some() != opening.current
                    || reading.next.is_some() != opening.next
            })
        {
            return Err(BooleanWhirError::ReadingShape { index });
        }
        let claims = Self::claims(openings)?;

        // The batch turns every reading about the bits into one claim about the packing.
        let readings = readings
            .iter()
            .map(|reading| (reading.current, reading.next))
            .collect::<Vec<_>>();
        claims
            .verify_readings(&proof.reduction, &readings, challenger)
            .map_err(BooleanWhirError::ReductionProof)
    }

    /// Open the bit witness with the readings every opening asks for, in one proof.
    ///
    /// No point needs prior transcript binding: each reduction binds its own.
    ///
    /// # Returns
    ///
    /// One set of readings per opening, in the order the openings were supplied.
    ///
    /// # Errors
    ///
    /// Before the transcript moves, an opening that names the wrong variables.
    ///
    /// An opening asking for no reading, or stepping within more rows than the witness has.
    #[allow(clippy::type_complexity)]
    pub fn open_readings(
        &self,
        prover_data: BooleanWhirData<F, EF, MT>,
        openings: &[BitOpening<EF>],
        challenger: &mut Challenger,
    ) -> Result<(Vec<BitReadings<EF>>, BooleanWhirProof<F, EF, MT>), BooleanWhirError> {
        let (readings, reduction, surviving_point) =
            self.switch(&prover_data, openings, challenger)?;

        // The surviving value crosses the wire twice, and the closing check is that the two agree.
        // The surviving point came out of the batch's rounds, so it is bound already.
        let opening = self
            .inner
            .open_at(
                prover_data,
                &self.protocol(1),
                &[surviving_point],
                challenger,
            )
            .map_err(BooleanWhirError::Commit)?;

        Ok((
            readings,
            BooleanWhirProof {
                reduction,
                opening: BooleanWhirOpening::Own(opening),
            },
        ))
    }

    /// Check one proof against the readings it claims at every opening.
    ///
    /// The commitment's binding is the caller's, replayed before this is reached.
    ///
    /// # Errors
    ///
    /// Before the transcript moves, a refused opening or a reading count that disagrees.
    ///
    /// After it, a false reading, a failed reduction, the commitment, or an unclosed claim.
    pub fn verify_readings(
        &self,
        commitment: &MT::Commitment,
        openings: &[BitOpening<EF>],
        readings: &[BitReadings<EF>],
        proof: &BooleanWhirProof<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> Result<(), BooleanWhirError> {
        let BooleanWhirOpening::Own(opening) = &proof.opening else {
            return Err(BooleanWhirError::OpeningKind);
        };
        let (surviving_point, surviving_value) =
            self.check_switch(openings, readings, proof, challenger)?;

        // One proximity opening pins the one surviving point to the committed polynomial.
        let evals = self
            .inner
            .verify_at(
                commitment,
                opening,
                &self.protocol(1),
                &[surviving_point],
                challenger,
            )
            .map_err(BooleanWhirError::Opening)?;

        // The batch closes against the value opened at its surviving point.
        match evals.as_slice() {
            [batch] if batch.current().first() == Some(&surviving_value) => Ok(()),
            _ => Err(BooleanWhirError::SurvivingClaim),
        }
    }

    /// Refuse a pair whose two commitments are not words of one code.
    ///
    /// The pair opening reads both trees under this commitment's schedule, so the second
    /// must have been encoded exactly as this one encodes.
    fn check_partner(&self, second: &Self) -> Result<(), BooleanWhirError> {
        let (a, b) = (&self.inner, &second.inner);
        if self.num_variables != second.num_variables
            || a.num_variables() != b.num_variables()
            || a.params().starting_log_inv_rate != b.params().starting_log_inv_rate
            || a.round_folding_factor(0) != b.round_folding_factor(0)
        {
            return Err(BooleanWhirError::PairShape {
                first: self.num_variables,
                second: second.num_variables,
            });
        }
        Ok(())
    }

    /// Open this witness and `second`'s, each with its own readings, under one proximity opening.
    ///
    /// ```text
    ///     switch      first's openings  ->  one claim about f at z_f
    ///     switch      second's openings ->  one claim about g at z_g
    ///     pair        f + gamma g at z_f and z_g, one WHIR run, both trees opened once
    /// ```
    ///
    /// # Errors
    ///
    /// As [`Self::open_readings`] for either side, and a partner of another code.
    #[allow(clippy::type_complexity)]
    pub fn open_readings_pair(
        &self,
        second: &Self,
        first_side: (BooleanWhirData<F, EF, MT>, &[BitOpening<EF>]),
        second_side: (&BooleanWhirData<F, EF, MT>, &[BitOpening<EF>]),
        challenger: &mut Challenger,
    ) -> Result<PairReadings<EF, BooleanWhirProof<F, EF, MT>>, BooleanWhirError> {
        self.check_partner(second)?;
        second.check_openings(second_side.1)?;
        let (first_readings, first_reduction, first_point) =
            self.switch(&first_side.0, first_side.1, challenger)?;
        let (second_readings, second_reduction, second_point) =
            second.switch(second_side.0, second_side.1, challenger)?;

        let pair = tracing::info_span!("pair opening")
            .in_scope(|| {
                self.inner.open_pair_at(
                    first_side.0,
                    second_side.0,
                    &self.protocol(1),
                    &[first_point, second_point],
                    challenger,
                )
            })
            .map_err(BooleanWhirError::Commit)?;

        Ok((
            (
                first_readings,
                BooleanWhirProof {
                    reduction: first_reduction,
                    opening: BooleanWhirOpening::Paired,
                },
            ),
            (
                second_readings,
                BooleanWhirProof {
                    reduction: second_reduction,
                    opening: BooleanWhirOpening::Pair(pair),
                },
            ),
        ))
    }

    /// Check what [`Self::open_readings_pair`] produced.
    ///
    /// # Errors
    ///
    /// As [`Self::verify_readings`] for either side, a partner of another code, or a pair whose
    /// opening sits on the wrong side.
    pub fn verify_readings_pair(
        &self,
        second: &Self,
        first_side: &ReadingsCheck<'_, EF, MT::Commitment, BooleanWhirProof<F, EF, MT>>,
        second_side: &ReadingsCheck<'_, EF, MT::Commitment, BooleanWhirProof<F, EF, MT>>,
        challenger: &mut Challenger,
    ) -> Result<(), BooleanWhirError> {
        self.check_partner(second)?;
        let (BooleanWhirOpening::Paired, BooleanWhirOpening::Pair(pair)) =
            (&first_side.proof.opening, &second_side.proof.opening)
        else {
            return Err(BooleanWhirError::OpeningKind);
        };
        second.check_openings(second_side.openings)?;
        let (first_point, first_value) = self.check_switch(
            first_side.openings,
            first_side.readings,
            first_side.proof,
            challenger,
        )?;
        let (second_point, second_value) = second.check_switch(
            second_side.openings,
            second_side.readings,
            second_side.proof,
            challenger,
        )?;

        let evals = self
            .inner
            .verify_pair_at(
                [first_side.commitment, second_side.commitment],
                pair,
                &self.protocol(1),
                &[first_point, second_point],
                challenger,
            )
            .map_err(BooleanWhirError::Opening)?;

        // Each side's batch closes against the value the pair bound at its surviving point.
        match evals.as_slice() {
            [first, second]
                if first.current().first() == Some(&first_value)
                    && second.current().first() == Some(&second_value) =>
            {
                Ok(())
            }
            _ => Err(BooleanWhirError::SurvivingClaim),
        }
    }

    /// Every labelled algebraic error one pair opening charges, one report per side.
    ///
    /// ```text
    ///     first side     the proximity argument at two surviving points, the pair's batching
    ///                    draw, and the first side's reductions
    ///     second side    the second side's reductions
    /// ```
    ///
    /// Both reports leave the same candidate count, the shared argument's.
    #[must_use]
    pub fn readings_pair_security(
        &self,
        second: &Self,
        first_side: (usize, bool),
        second_side: (usize, bool),
    ) -> Option<(PrescribedOpeningSecurity, PrescribedOpeningSecurity)> {
        self.check_partner(second).ok()?;
        let mut first = self.inner.prescribed_security(&self.protocol(2))?;
        first.terms.push(SecurityTerm::new(
            PAIR_BATCHING_LABEL,
            ErrorBits::from_log2(pair_batching_error(&self.inner)),
        ));
        let mut second_report = PrescribedOpeningSecurity {
            terms: Vec::new(),
            log2_max_candidates: first.log2_max_candidates,
        };
        for ((num_claims, successor_tensors), report) in
            [(first_side, &mut first), (second_side, &mut second_report)]
        {
            for term in
                Self::reduction_terms(self.inner.num_variables(), num_claims, successor_tensors)
            {
                report.charge_reduction(term);
            }
        }
        Some((first, second_report))
    }

    /// The reductions one batched ring switch of this many claims draws.
    fn reduction_terms(
        packed_variables: usize,
        num_claims: usize,
        successor_tensors: bool,
    ) -> Vec<SecurityTerm> {
        // The tensor alone, or the tensor with carry and last.
        let num_tensors = if successor_tensors { 3 } else { 1 };
        let mut terms = vec![bit_ring_switch_tensors_term(
            num_claims.min(1),
            num_tensors,
            BitRingSwitch::<F, EF>::BATCHED,
            packed_variables,
            EF::bits(),
        )];
        // A single claim draws no lambda, so its report carries no batching term.
        if num_claims > 1 {
            terms.push(bit_ring_switch_claim_batching_term(num_claims, EF::bits()));
        }
        terms
    }
}

impl<F, EF, Dft, MT, Challenger> BooleanMultilinearPcs<EF, Challenger>
    for BooleanWhirPcs<F, EF, Dft, MT, Challenger>
where
    F: Field + TranscriptField + TowerLevel + Coordinates + Ord + Send + Sync,
    EF: BitChallengeField<F>,
    Dft: WhirDomain<F, EF>,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<MT::Commitment>,
{
    fn observe_commitment(&self, commitment: &Self::Commitment, challenger: &mut Challenger) {
        p3_commit::MultilinearPcs::<EF, Challenger>::observe_commitment(
            &self.inner,
            commitment,
            challenger,
        );
    }

    fn commit_bits<U: Underlier>(
        &self,
        bits: &[PackedGf2<U>],
        challenger: &mut Challenger,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error> {
        // The packing is one copy of the bits, so the witness is never swept for arithmetic.
        let stack = PackedStack::<PackedGf2<U>, F>::from_columns(&[bits])
            .map_err(BooleanWhirError::Packing)?;
        self.commit_column(stack.into_poly().into_evals(), challenger)
    }

    fn commit_packed(
        &self,
        packed: PackedWords<F>,
        challenger: &mut Challenger,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error> {
        self.commit_column(packed.into_elements(), challenger)
    }

    fn packed_words(words: usize) -> Option<PackedWords<F>> {
        PackedWords::zeroed(words)
    }

    fn open_readings(
        &self,
        prover_data: Self::ProverData,
        openings: &[BitOpening<EF>],
        challenger: &mut Challenger,
    ) -> Result<(Vec<BitReadings<EF>>, Self::Proof), Self::Error> {
        Self::open_readings(self, prover_data, openings, challenger)
    }

    fn verify_readings(
        &self,
        commitment: &Self::Commitment,
        openings: &[BitOpening<EF>],
        readings: &[BitReadings<EF>],
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        Self::verify_readings(self, commitment, openings, readings, proof, challenger)
    }

    fn readings_security(
        &self,
        num_claims: usize,
        successor_tensors: bool,
    ) -> Option<PrescribedOpeningSecurity> {
        Self::readings_security(self, num_claims, successor_tensors)
    }

    fn open_readings_pair(
        &self,
        second: &Self,
        first_side: (Self::ProverData, &[BitOpening<EF>]),
        second_side: (&Self::ProverData, &[BitOpening<EF>]),
        challenger: &mut Challenger,
    ) -> Result<PairReadings<EF, Self::Proof>, Self::Error> {
        Self::open_readings_pair(self, second, first_side, second_side, challenger)
    }

    fn verify_readings_pair(
        &self,
        second: &Self,
        first_side: ReadingsCheck<'_, EF, Self::Commitment, Self::Proof>,
        second_side: ReadingsCheck<'_, EF, Self::Commitment, Self::Proof>,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        Self::verify_readings_pair(self, second, &first_side, &second_side, challenger)
    }

    fn readings_pair_security(
        &self,
        second: &Self,
        first_side: (usize, bool),
        second_side: (usize, bool),
    ) -> Option<(PrescribedOpeningSecurity, PrescribedOpeningSecurity)> {
        Self::readings_pair_security(self, second, first_side, second_side)
    }

    fn open_at_points(
        &self,
        prover_data: Self::ProverData,
        points: &[Point<EF>],
        challenger: &mut Challenger,
    ) -> Result<(Vec<EF>, Self::Proof), Self::Error> {
        let (readings, proof) =
            self.open_readings(prover_data, &Self::current_openings(points), challenger)?;
        let values = readings
            .into_iter()
            .map(|reading| {
                reading
                    .current
                    .expect("an opening asking for the current reading carries it")
            })
            .collect();
        Ok((values, proof))
    }

    fn verify_at_points(
        &self,
        commitment: &Self::Commitment,
        points: &[Point<EF>],
        values: &[EF],
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        let readings: Vec<BitReadings<EF>> = values
            .iter()
            .map(|&value| BitReadings {
                current: Some(value),
                next: None,
            })
            .collect();
        self.verify_readings(
            commitment,
            &Self::current_openings(points),
            &readings,
            proof,
            challenger,
        )
    }
}

impl<F, EF, Dft, MT, Challenger> BooleanBackend<EF> for BooleanWhirPcs<F, EF, Dft, MT, Challenger>
where
    F: Field + TranscriptField + TowerLevel + Coordinates + Ord + Send + Sync,
    EF: BitCoordinates + ExtensionField<F>,
    Dft: WhirDomain<F, EF>,
    MT: Mmcs<F>,
{
    type Val = F;
    type Commitment = MT::Commitment;
    type ProverData = BooleanWhirData<F, EF, MT>;
    type Proof = BooleanWhirProof<F, EF, MT>;
    type Error = BooleanWhirError;

    fn num_variables(&self) -> usize {
        self.num_variables
    }
}
