//! One WHIR opening for two commitments of the same shape.
//!
//! ```text
//!     f, g          two committed polynomials, one tree each, one layout shape
//!     z_f, z_g      the point each is opened at
//!     send          f(z_f), g(z_g), and the cross values g(z_f), f(z_g)
//!     draw          gamma in the committed field, after a grind
//!     prove         F = f + gamma g at z_f and z_g, by one WHIR run
//! ```
//!
//! The run commits nothing new for `F`.
//!
//! Its first round's queries open both trees at the same positions, and the verifier
//! combines the two rows by `[1, gamma]`, which is the row `F`'s codeword holds there.
//!
//! Every later round, the terminal round and the final polynomial are paid once.
//!
//! # Soundness
//!
//! All four values are bound before `gamma` is drawn.
//!
//! A pair not jointly close to the code leaves `F` close with probability at most the
//! proximity-gap error of a two-function batch, charged by [`pair_batching_error`].
//!
//! A pair that is close is pinned at both points by `F`'s two claims, each up to `1/|F|`.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::split_eq::SplitEq;
use p3_sumcheck::layout::{DeviceSuffix, Layout, Table, Verifier};
use p3_sumcheck::{OpeningEvals, OpeningProtocol, OpeningRequest, TableShape};
use serde::{Deserialize, Serialize};

use super::adapter::WhirProverData;
use super::device::{PairDevices, SingleDevices};
use super::prover::WhirProver;
use super::verifier::errors::VerifierError;
use super::verifier::{BaseCommitments, WhirVerifier};
use crate::WhirConfigError;
use crate::domain::WhirDomain;
use crate::parameters::WhirConfig;
use crate::pcs::proof::{PcsProof, WhirProof};

/// One opening of two commitments, each at its own point.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, MT::MultiProof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, MT::MultiProof: Deserialize<'de>"
))]
pub struct PairProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Each polynomial at its own point: `f(z_f)`, then `g(z_g)`.
    pub evals: Vec<OpeningEvals<EF>>,
    /// Each polynomial at the other's point: `g(z_f)`, then `f(z_g)`.
    pub cross: Vec<OpeningEvals<EF>>,
    /// Grinding witness guarding the draw of `gamma`.
    pub pow_witness: F,
    /// The run proving `f + gamma g` at both points.
    pub whir: WhirProof<F, EF, MT>,
}

/// The one request a pair opening serves, against each commitment in turn.
fn single_request(protocol: &OpeningProtocol) -> (usize, &OpeningRequest) {
    let mut openings = protocol.iter_openings();
    let opening = openings
        .next()
        .expect("a pair opening opens each commitment once");
    assert!(
        openings.next().is_none(),
        "a pair opening opens each commitment once"
    );
    opening
}

/// The current-row columns `request` names, each evaluated at `point`.
///
/// The multilinear extension of a column over its own rows, as every layout claims it, read
/// off the table without recording a claim.
fn column_evals<F: Field, EF: ExtensionField<F>>(
    table: &Table<F>,
    request: &OpeningRequest,
    point: &Point<EF>,
) -> OpeningEvals<EF> {
    let eq = SplitEq::<F, EF>::new_packed(point, EF::ONE);
    OpeningEvals::new(
        request
            .current()
            .iter()
            .map(|&column| eq.eval_base(table.poly(column)))
            .collect(),
        Vec::new(),
    )
}

/// `a + gamma b`, coordinate by coordinate, over both views.
fn combine<F: Field, EF: ExtensionField<F>>(
    a: &OpeningEvals<EF>,
    b: &OpeningEvals<EF>,
    gamma: F,
) -> OpeningEvals<EF> {
    let mix = |a: &[EF], b: &[EF]| {
        a.iter()
            .zip(b)
            .map(|(&a, &b)| a + b * gamma)
            .collect::<Vec<_>>()
    };
    OpeningEvals::new(mix(a.current(), b.current()), mix(a.next(), b.next()))
}

/// Bind the four values ahead of the grind and the draw of `gamma`.
fn bind_values<F, EF, Challenger>(
    evals: &[OpeningEvals<EF>],
    cross: &[OpeningEvals<EF>],
    challenger: &mut Challenger,
) where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    for batch in evals.iter().chain(cross) {
        challenger.observe_algebra_slice(batch.current());
        challenger.observe_algebra_slice(batch.next());
    }
}

/// Log2 of the error of folding two committed functions under one draw of `gamma`.
///
/// The draw lives in the committed field `F`, so the field term is `F`'s.
///
/// The grind before it is the starting fold's, which prices the same gap at the same
/// domain, so the term lands where that fold's own term lands when `F` is `EF`.
#[must_use]
pub fn pair_batching_error<EF, F, Challenger>(config: &WhirConfig<EF, F, Challenger>) -> f64
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let field_bits = F::bits().saturating_sub(1);
    config.soundness_type.prox_gaps_error(
        config.num_variables,
        config.starting_log_inv_rate,
        field_bits,
        2,
    ) + config.starting_folding_pow_bits as f64
}

impl<EF, F, Dft, MT, Challenger, L> WhirProver<EF, F, Dft, MT, Challenger, L>
where
    F: Field + TranscriptField + Ord,
    EF: ExtensionField<F>,
    Dft: WhirDomain<F, EF>,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<MT::Commitment>,
    L: Layout<F, EF>,
{
    /// Open two commitments of this scheme's shape, `first` at `points[0]` and `second`
    /// at `points[1]`, in one proof.
    ///
    /// Neither commitment is bound here: each was bound when it was made.
    ///
    /// The second is read in place, never copied: a commitment reused across proofs, as a
    /// preprocessed one is, keeps its one codeword and its one set of tables.
    ///
    /// # Errors
    ///
    /// Returns an error when the claims exceed the configured budget.
    ///
    /// # Panics
    ///
    /// When the protocol asks for anything but one opening of current-row columns, or the two
    /// layouts differ.
    pub fn open_pair_at(
        &self,
        first: WhirProverData<F, EF, MT, L>,
        second: &WhirProverData<F, EF, MT, L>,
        protocol: &OpeningProtocol,
        points: &[Point<EF>; 2],
        challenger: &mut Challenger,
    ) -> Result<PairProof<F, EF, MT>, WhirConfigError> {
        self.open_pair(first, second, protocol, points, None, challenger)
    }

    /// [`Self::open_pair_at`] with the dense work on `device`, which holds `first`'s message.
    ///
    /// The proof is the one [`Self::open_pair_at`] makes.
    ///
    /// # Errors
    ///
    /// As [`Self::open_pair_at`].
    ///
    /// # Panics
    ///
    /// As [`Self::open_pair_at`], and unless each commitment is one whole-space column.
    pub fn open_pair_at_on(
        &self,
        first: WhirProverData<F, EF, MT, L>,
        second: &WhirProverData<F, EF, MT, L>,
        protocol: &OpeningProtocol,
        points: &[Point<EF>; 2],
        device: &dyn PairDevices<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> Result<PairProof<F, EF, MT>, WhirConfigError> {
        self.open_pair(first, second, protocol, points, Some(device), challenger)
    }

    /// The opening [`PrescribedPointPcs::open_at`](p3_sumcheck::PrescribedPointPcs::open_at) makes of
    /// one commitment at one point, every round on `device`, which is handed the message.
    ///
    /// # Errors
    ///
    /// Returns an error when the claims exceed the configured budget.
    ///
    /// # Panics
    ///
    /// Unless the protocol asks for one opening of the current row of one whole-space column.
    pub fn open_at_on(
        &self,
        prover_data: WhirProverData<F, EF, MT, L>,
        protocol: &OpeningProtocol,
        points: &[Point<EF>],
        device: &dyn SingleDevices<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> Result<PcsProof<F, EF, MT>, WhirConfigError> {
        self.config.validate_initial_claims(
            protocol
                .checked_num_claims()
                .and_then(|n| n.checked_add(self.commitment_ood_samples))
                .ok_or(WhirConfigError::InitialClaimCountOverflow)?,
        )?;
        let (table_idx, request) = single_request(protocol);
        let [point] = points else {
            panic!("one point for the one opening");
        };
        assert!(
            request.current() == [0] && request.next().is_empty(),
            "a device opening reads the one column's current row"
        );
        let WhirProverData {
            layout: f,
            merkle_data,
            ..
        } = prover_data;
        let num_variables = f.num_variables();
        assert_eq!(
            protocol.table_shapes(),
            [TableShape::new(num_variables, 1)],
            "a device opens one whole-space column"
        );

        let resident = device.holds_commitment();
        let mut session = device.single(f.table(0).iter_polys().next().expect("one column"));
        drop(f);
        let mut layout = DeviceSuffix::new(num_variables, self.round_folding_factor(0));
        let initial_ood_answers = (0..self.commitment_ood_samples)
            .map(|_| layout.add_virtual_eval(session.stacked(), challenger))
            .collect::<Vec<_>>();
        let evals = OpeningEvals::new(vec![session.stacked().eval(point)], Vec::new());
        layout.eval_at_known(table_idx, request, point, &evals, challenger);
        let num_claims = layout.num_claims();
        let whir = self.prove_single_on(
            initial_ood_answers,
            challenger,
            num_claims,
            |sumcheck, challenger, device| {
                layout.into_rounds(
                    device.stacked(),
                    sumcheck,
                    self.starting_folding_pow_bits,
                    challenger,
                )
            },
            &mut *session,
            (!resident).then_some(merkle_data),
            protocol.num_openings(),
        )?;
        Ok(PcsProof {
            whir,
            evals: vec![evals],
        })
    }

    fn open_pair(
        &self,
        first: WhirProverData<F, EF, MT, L>,
        second: &WhirProverData<F, EF, MT, L>,
        protocol: &OpeningProtocol,
        points: &[Point<EF>; 2],
        device: Option<&dyn PairDevices<F, EF, MT>>,
        challenger: &mut Challenger,
    ) -> Result<PairProof<F, EF, MT>, WhirConfigError> {
        let (table_idx, request) = single_request(protocol);
        let shapes = protocol.table_shapes();
        assert_eq!(first.layout.table_shapes(), shapes);
        assert_eq!(second.layout.table_shapes(), shapes);
        self.config.validate_initial_claims(
            request
                .len()
                .checked_mul(2)
                .and_then(|n| n.checked_add(self.commitment_ood_samples))
                .ok_or(WhirConfigError::InitialClaimCountOverflow)?,
        )?;

        assert!(
            request.next().is_empty(),
            "a pair opening reads the current row only"
        );

        let WhirProverData {
            layout: f,
            merkle_data: f_data,
            ..
        } = first;
        let g = &second.layout;

        let mut session = device.map(|device| {
            let whole = [TableShape::new(f.num_variables(), 1)];
            assert_eq!(shapes, whole, "a device opens one whole-space column");
            let second = g.table(0).iter_polys().next().expect("one column");
            if f.table(0).is_held() {
                return device.pair_held(second);
            }
            let first = f.table(0).iter_polys().next().expect("one column");
            device.pair(first, second)
        });
        let (evals, cross) = tracing::info_span!("pair values").in_scope(|| {
            if let Some(session) = session.as_mut() {
                let one = |value: EF| OpeningEvals::new(vec![value], Vec::new());
                let [f0, g1, g0, f1] = session.values(points);
                return (vec![one(f0), one(g1)], vec![one(g0), one(f1)]);
            }
            let at = |layout: &L, point: &Point<EF>| {
                column_evals(layout.table(table_idx), request, point)
            };
            let evals = vec![at(&f, &points[0]), at(g, &points[1])];
            let cross = vec![at(g, &points[0]), at(&f, &points[1])];
            (evals, cross)
        });

        bind_values::<F, EF, _>(&evals, &cross, challenger);
        let bits = self.starting_folding_pow_bits;
        let pow_witness = if bits == 0 {
            F::ZERO
        } else {
            challenger.grind(bits)
        };
        let gamma: F = challenger.sample();

        if let Some(mut session) = session {
            let num_variables = f.num_variables();
            drop(f);
            drop(f_data);
            tracing::info_span!("pair combine").in_scope(|| session.combine(gamma));
            let device = session.rounds();
            let mut layout = DeviceSuffix::new(num_variables, self.round_folding_factor(0));
            let initial_ood_answers = (0..self.commitment_ood_samples)
                .map(|_| layout.add_virtual_eval(device.stacked(), challenger))
                .collect::<Vec<_>>();
            for (point, (own, other)) in points
                .iter()
                .zip([(&evals[0], &cross[0]), (&cross[1], &evals[1])])
            {
                let combined = combine(own, other, gamma);
                layout.eval_at_known(table_idx, request, point, &combined, challenger);
            }
            let num_claims = layout.num_claims();
            let whir = self.prove_pair_on(
                initial_ood_answers,
                challenger,
                num_claims,
                |sumcheck, challenger, device| {
                    layout.into_rounds(
                        device.stacked(),
                        sumcheck,
                        self.starting_folding_pow_bits,
                        challenger,
                    )
                },
                device,
                second.merkle_data.clone(),
                gamma,
                2,
            )?;
            return Ok(PairProof {
                evals,
                cross,
                pow_witness,
                whir,
            });
        }

        let tables = tracing::info_span!("pair combine").in_scope(|| {
            (0..shapes.len())
                .map(|id| {
                    let (a, b) = (f.table(id), g.table(id));
                    let width = 1 << a.shape().num_variables();
                    let values = a
                        .iter_polys()
                        .zip(b.iter_polys())
                        .flat_map(|(a, b)| a.iter().zip(b).map(|(&a, &b)| a + b * gamma))
                        .collect::<Vec<_>>();
                    Table::new(RowMajorMatrix::new(values, width))
                })
                .collect::<Vec<_>>()
        });
        drop(f);
        let mut layout = L::from_witness(L::new_witness(tables, self.round_folding_factor(0)));

        let initial_ood_answers = (0..self.commitment_ood_samples)
            .map(|_| layout.add_virtual_eval(challenger))
            .collect::<Vec<_>>();
        for (point, (own, other)) in points
            .iter()
            .zip([(&evals[0], &cross[0]), (&cross[1], &evals[1])])
        {
            let combined = layout.eval_at(table_idx, request, point, challenger);
            debug_assert_eq!(combined, combine(own, other, gamma));
        }

        let whir = self.prove_batched(
            initial_ood_answers,
            challenger,
            layout,
            vec![f_data, Arc::clone(&second.merkle_data)],
            vec![F::ONE, gamma],
            2,
        )?;

        Ok(PairProof {
            evals,
            cross,
            pow_witness,
            whir,
        })
    }

    /// Verify one pair opening of two commitments, each at its own point.
    ///
    /// # Returns
    ///
    /// The values bound to each commitment at its own point: `f(z_f)`, then `g(z_g)`.
    ///
    /// # Errors
    ///
    /// A proof of the wrong shape before the transcript moves; after it, a failed grind,
    /// a claim of `f + gamma g` the run does not close, or a failed run.
    pub fn verify_pair_at(
        &self,
        commitments: [&MT::Commitment; 2],
        proof: &PairProof<F, EF, MT>,
        protocol: &OpeningProtocol,
        points: &[Point<EF>; 2],
        challenger: &mut Challenger,
    ) -> Result<Vec<OpeningEvals<EF>>, VerifierError> {
        let (table_idx, request) = single_request(protocol);
        if proof.evals.len() != 2 || proof.cross.len() != 2 {
            return Err(VerifierError::OpeningBatchCountMismatch {
                expected: 2,
                actual: proof.evals.len().min(proof.cross.len()),
            });
        }
        if let Some(batch) = proof
            .evals
            .iter()
            .chain(&proof.cross)
            .find(|batch| !request.has_same_shape(*batch))
        {
            return Err(VerifierError::OpeningBatchSizeMismatch {
                table_idx,
                expected: request.len(),
                actual: batch.len(),
            });
        }
        if proof.whir.initial_ood_answers.len() != self.commitment_ood_samples {
            return Err(VerifierError::InitialOodAnswerCountMismatch {
                expected: self.commitment_ood_samples,
                actual: proof.whir.initial_ood_answers.len(),
            });
        }
        let bits = self.starting_folding_pow_bits;
        if bits == 0 && proof.pow_witness != F::ZERO {
            return Err(VerifierError::NonCanonicalPairPowWitness);
        }
        self.config.validate_initial_claims(
            request
                .len()
                .saturating_mul(2)
                .saturating_add(self.commitment_ood_samples),
        )?;

        bind_values::<F, EF, _>(&proof.evals, &proof.cross, challenger);
        if bits != 0 && !challenger.check_witness(bits, proof.pow_witness) {
            return Err(VerifierError::InvalidPairPowWitness { bits });
        }
        let gamma: F = challenger.sample();

        let mut layout = Verifier::<F, EF>::new(&protocol.table_shapes(), L::strategy());
        for &eval in &proof.whir.initial_ood_answers {
            layout.add_virtual_eval(eval, challenger);
        }
        for (point, (own, other)) in points.iter().zip([
            (&proof.evals[0], &proof.cross[0]),
            (&proof.cross[1], &proof.evals[1]),
        ]) {
            layout.add_claim_at(
                table_idx,
                request,
                point,
                &combine(own, other, gamma),
                challenger,
            )?;
        }

        let roots = [commitments[0].clone(), commitments[1].clone()];
        let coefficients = [F::ONE, gamma];
        let verifier = WhirVerifier::new(&self.config, &self.dft, &self.mmcs, L::variable_order());
        let _ = verifier.verify_batched(
            &proof.whir,
            challenger,
            BaseCommitments {
                roots: &roots,
                coefficients: &coefficients,
            },
            2,
            &layout,
        )?;

        Ok(proof.evals.clone())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_commit::MultilinearPcs;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::point::Point;
    use p3_sumcheck::layout::{Layout, PrefixProver, SuffixProver, Table, observe_commitment};
    use p3_sumcheck::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::PairProof;
    use crate::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
    use crate::pcs::prover::WhirProver;
    use crate::pcs::tests::challenger;
    use crate::pcs::verifier::errors::VerifierError;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
    type Packed = <F as p3_field::Field>::Packing;
    type Mmcs = MerkleTreeMmcs<Packed, Packed, Hash, Compress, 2, 8>;
    type Challenger = p3_challenger::DuplexChallenger<F, Perm, 16, 8>;
    type Pcs<L> = WhirProver<EF, F, Radix2DFTSmallBatch<F>, Mmcs, Challenger, L>;
    type Commitment = <Mmcs as p3_commit::Mmcs<F>>::Commitment;

    /// How a run departs from the honest one.
    #[derive(Clone, Copy)]
    enum Tamper {
        None,
        OwnValue,
        CrossValue,
        SwappedCommitments,
        WitnessOfAnotherPolynomial,
    }

    fn pcs<L: Layout<F, EF>>(num_variables: usize, pow_bits: usize) -> Pcs<L> {
        let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
        let mmcs = Mmcs::new(Hash::new(perm.clone()), Compress::new(perm), 0);
        let folding_factor = FoldingFactor::Constant(2);
        let schedule = folding_factor
            .compute_folding_schedule(num_variables)
            .unwrap();
        let mut rate = 1;
        let round_log_inv_rates = schedule[..schedule.len() - 1]
            .iter()
            .map(|folding| {
                rate += folding - 1;
                rate
            })
            .collect();
        let config = WhirConfig::new(
            num_variables,
            ProtocolParameters {
                security_level: 32,
                pow_bits,
                round_log_inv_rates,
                folding_factor,
                soundness_type: SecurityAssumption::CapacityBound,
                starting_log_inv_rate: 1,
            },
        )
        .unwrap();
        Pcs::new(config, Radix2DFTSmallBatch::default(), mmcs)
    }

    fn run<L: Layout<F, EF>>(pow_bits: usize, tamper: Tamper) -> Result<(), VerifierError> {
        let spec = TableSpec::new(
            TableShape::new(8, 2),
            vec![OpeningBatch::new(vec![0, 1], Vec::new())],
        );
        let protocol = OpeningProtocol::new(vec![spec.clone()]);
        let pcs = pcs::<L>(9, pow_bits);
        let folding = pcs.round_folding_factor(0);
        let point = |seed: u64| Point::new((0..8).map(|c| EF::from_u64(seed + 3 * c)).collect());
        let points = [point(7), point(11)];

        let mut prover = challenger();
        let commit = |seed: u64, challenger: &mut Challenger| {
            let table = Table::rand(&mut SmallRng::seed_from_u64(seed), 2, 8);
            <Pcs<L> as MultilinearPcs<EF, Challenger>>::commit(
                &pcs,
                L::new_witness(vec![table], folding),
                challenger,
            )
            .unwrap()
        };
        let (first, first_data) = commit(1, &mut prover);
        let (second, second_data) = commit(2, &mut prover);
        let unrelated = matches!(tamper, Tamper::WitnessOfAnotherPolynomial)
            .then(|| commit(3, &mut challenger()).1);
        let mut proof: PairProof<F, EF, Mmcs> = pcs
            .open_pair_at(
                first_data,
                &unrelated.unwrap_or(second_data),
                &protocol,
                &points,
                &mut prover,
            )
            .unwrap();

        let bump = |batch: &mut OpeningBatch<EF>| {
            let mut current = batch.current().to_vec();
            current[0] += EF::ONE;
            *batch = OpeningBatch::new(current, batch.next().to_vec());
        };
        match tamper {
            Tamper::OwnValue => bump(&mut proof.evals[1]),
            Tamper::CrossValue => bump(&mut proof.cross[0]),
            _ => {}
        }
        let roots: [&Commitment; 2] = if matches!(tamper, Tamper::SwappedCommitments) {
            [&second, &first]
        } else {
            [&first, &second]
        };

        let mut verifier = challenger();
        observe_commitment::<F, _, _>(&mut verifier, first.clone());
        observe_commitment::<F, _, _>(&mut verifier, second.clone());
        let evals = pcs.verify_pair_at(roots, &proof, &protocol, &points, &mut verifier)?;
        assert_eq!(evals, proof.evals);
        Ok(())
    }

    fn all_tampers<L: Layout<F, EF>>(pow_bits: usize) {
        run::<L>(pow_bits, Tamper::None).expect("an honest pair opening verifies");
        for tamper in [
            Tamper::OwnValue,
            Tamper::CrossValue,
            Tamper::SwappedCommitments,
            Tamper::WitnessOfAnotherPolynomial,
        ] {
            assert!(run::<L>(pow_bits, tamper).is_err());
        }
    }

    #[test]
    fn a_pair_opening_verifies_and_rejects_every_tamper() {
        all_tampers::<PrefixProver<F, EF>>(0);
        all_tampers::<SuffixProver<F, EF>>(0);
    }

    #[test]
    fn a_ground_pair_opening_verifies() {
        all_tampers::<SuffixProver<F, EF>>(4);
    }

    #[test]
    fn a_pair_opening_shares_every_round_after_the_first() {
        let spec = TableSpec::new(
            TableShape::new(8, 2),
            vec![OpeningBatch::new(vec![0], Vec::new())],
        );
        let protocol = OpeningProtocol::new(vec![spec]);
        let pcs = pcs::<SuffixProver<F, EF>>(9, 0);
        let folding = pcs.round_folding_factor(0);
        let mut prover = challenger();
        let mut commit = |seed: u64| {
            let table = Table::rand(&mut SmallRng::seed_from_u64(seed), 2, 8);
            <Pcs<SuffixProver<F, EF>> as MultilinearPcs<EF, Challenger>>::commit(
                &pcs,
                SuffixProver::<F, EF>::new_witness(vec![table], folding),
                &mut prover,
            )
            .unwrap()
            .1
        };
        let (a, b) = (commit(1), commit(2));
        let point = Point::new((0..8).map(|c| EF::from_u64(5 + c)).collect());
        let proof = pcs
            .open_pair_at(a, &b, &protocol, &[point.clone(), point], &mut challenger())
            .unwrap();
        assert_eq!(proof.whir.rounds.len(), pcs.n_rounds());
        assert!(matches!(
            proof.whir.rounds[0].openings,
            crate::pcs::proof::QueryOpenings::Batched(_)
        ));
    }

    /// The device protocol played in host memory, one dense vector per side, every round's
    /// codeword committed with the scheme's own encoder and tree.
    struct HostPair<'a> {
        pcs: &'a Pcs<SuffixProver<F, EF>>,
        f: Vec<F>,
        g: Vec<F>,
        e: Vec<EF>,
        w: Vec<EF>,
        trees: Vec<Option<crate::ExtProverData<F, EF, Mmcs>>>,
    }

    struct HostDevices<'a>(&'a Pcs<SuffixProver<F, EF>>);

    impl crate::PairDevices<F, EF, Mmcs> for HostDevices<'_> {
        fn pair<'a>(
            &'a self,
            first: &[F],
            second: &[F],
        ) -> alloc::boxed::Box<dyn crate::PairDevice<F, EF, Mmcs> + 'a> {
            alloc::boxed::Box::new(HostPair {
                pcs: self.0,
                f: first.to_vec(),
                g: second.to_vec(),
                e: Vec::new(),
                w: Vec::new(),
                trees: Vec::new(),
            })
        }

        fn packing_rounds(&self) -> &dyn p3_sumcheck::ring_switch::bits::PackingRoundsDevice {
            &p3_sumcheck::ring_switch::bits::HostPackingRounds
        }
    }

    impl crate::SingleDevices<F, EF, Mmcs> for HostDevices<'_> {
        fn packing_rounds(&self) -> &dyn p3_sumcheck::ring_switch::bits::PackingRoundsDevice {
            &p3_sumcheck::ring_switch::bits::HostPackingRounds
        }

        fn single<'a>(
            &'a self,
            message: &[F],
        ) -> alloc::boxed::Box<dyn crate::RoundDevice<F, EF, Mmcs> + 'a> {
            alloc::boxed::Box::new(HostPair {
                pcs: self.0,
                f: message.to_vec(),
                g: Vec::new(),
                e: message.iter().map(|&x| EF::from(x)).collect(),
                w: Vec::new(),
                trees: Vec::new(),
            })
        }
    }

    fn dot(values: impl Iterator<Item = EF>, point: &Point<EF>) -> EF {
        let eq = p3_multilinear_util::poly::Poly::new_from_point(point.as_slice(), EF::ONE);
        values.zip(eq.as_slice()).map(|(v, &q)| v * q).sum()
    }

    impl crate::PairDevice<F, EF, Mmcs> for HostPair<'_> {
        fn values(&mut self, points: &[Point<EF>; 2]) -> [EF; 4] {
            let at = |v: &[F], p: &Point<EF>| dot(v.iter().map(|&x| EF::from(x)), p);
            [
                at(&self.f, &points[0]),
                at(&self.g, &points[1]),
                at(&self.g, &points[0]),
                at(&self.f, &points[1]),
            ]
        }

        fn combine(&mut self, gamma: F) {
            self.e = self
                .f
                .iter()
                .zip(&self.g)
                .map(|(&a, &b)| EF::from(a + b * gamma))
                .collect();
        }

        fn rounds(&mut self) -> &mut dyn crate::RoundDevice<F, EF, Mmcs> {
            self
        }
    }

    impl crate::RoundDevice<F, EF, Mmcs> for HostPair<'_> {
        fn stacked(&mut self) -> &mut dyn p3_sumcheck::layout::StackedDevice<F, EF> {
            self
        }

        fn commit(&mut self, folding: usize, log_inv_rate: usize) -> (Commitment, usize) {
            use p3_commit::Mmcs as _;

            use crate::domain::WhirDomain as _;

            let width = 1 << folding;
            let height = (self.e.len() >> folding) << log_inv_rate;
            let mut values = EF::zero_vec(height * width);
            values[..self.e.len()].copy_from_slice(&self.e);
            let encoded = self.pcs.dft.encode_extension_batch_padded(
                p3_matrix::dense::RowMajorMatrix::new(values, width),
                log_inv_rate,
            );
            let (root, data) = self.pcs.extension_mmcs.commit_matrix(encoded);
            self.trees.push(Some(data));
            (root, self.trees.len() - 1)
        }

        fn tree(&mut self, handle: usize, _indices: &[usize]) -> crate::ExtProverData<F, EF, Mmcs> {
            self.trees[handle]
                .take()
                .expect("each codeword is opened once")
        }

        fn first_tree(&mut self, _indices: &[usize]) -> crate::BaseProverData<F, Mmcs> {
            let folding = self.pcs.round_folding_factor(0);
            let table = Table::new(p3_matrix::dense::RowMajorMatrix::new(
                self.f.clone(),
                self.f.len(),
            ));
            let data = <Pcs<SuffixProver<F, EF>> as MultilinearPcs<EF, Challenger>>::commit(
                self.pcs,
                SuffixProver::<F, EF>::new_witness(vec![table], folding),
                &mut challenger(),
            )
            .unwrap()
            .1
            .merkle_data;
            alloc::sync::Arc::into_inner(data).expect("a fresh commit holds its only reference")
        }

        fn constrain(&mut self, eq: &[(Point<EF>, EF)], select: &[(Point<F>, EF)]) {
            for (point, scale) in eq {
                let eq = p3_multilinear_util::poly::Poly::new_from_point(point.as_slice(), *scale);
                for (w, &q) in self.w.iter_mut().zip(eq.as_slice()) {
                    *w += q;
                }
            }
            let n = self.w.len().trailing_zeros() as usize;
            for (point, scale) in select {
                assert_eq!(point.num_variables(), n);
                for (b, w) in self.w.iter_mut().enumerate() {
                    let weight = (0..n)
                        .filter(|i| b >> i & 1 == 1)
                        .fold(*scale, |acc, i| acc * point.as_slice()[n - 1 - i]);
                    *w += weight;
                }
            }
        }
    }

    impl p3_sumcheck::layout::StackedDevice<F, EF> for HostPair<'_> {
        fn eval(&mut self, point: &Point<EF>) -> EF {
            dot(self.e.iter().copied(), point)
        }

        fn weigh(&mut self, claims: &[(Point<EF>, EF)]) {
            self.w = vec![EF::ZERO; self.e.len()];
            for (point, scale) in claims {
                let eq = p3_multilinear_util::poly::Poly::new_from_point(point.as_slice(), *scale);
                for (w, &q) in self.w.iter_mut().zip(eq.as_slice()) {
                    *w += q;
                }
            }
        }

        fn round(&mut self) -> (EF, EF) {
            let mut c0 = EF::ZERO;
            let mut c_inf = EF::ZERO;
            for (e, w) in self.e.chunks(2).zip(self.w.chunks(2)) {
                c0 += e[0] * w[0];
                c_inf += (e[1] - e[0]) * (w[1] - w[0]);
            }
            (c0, c_inf)
        }

        fn fold(&mut self, r: EF) {
            let bind = |v: &[EF]| v.chunks(2).map(|p| p[0] + r * (p[1] - p[0])).collect();
            self.e = bind(&self.e);
            self.w = bind(&self.w);
        }

        fn residual(
            &mut self,
        ) -> (
            p3_multilinear_util::poly::Poly<EF>,
            p3_multilinear_util::poly::Poly<EF>,
        ) {
            (
                p3_multilinear_util::poly::Poly::new(self.e.clone()),
                p3_multilinear_util::poly::Poly::new(self.w.clone()),
            )
        }
    }

    #[test]
    fn a_device_pair_opening_is_the_host_one() {
        for pow_bits in [0, 4] {
            let spec = TableSpec::new(
                TableShape::new(9, 1),
                vec![OpeningBatch::new(vec![0], Vec::new())],
            );
            let protocol = OpeningProtocol::new(vec![spec]);
            let pcs = pcs::<SuffixProver<F, EF>>(9, pow_bits);
            let folding = pcs.round_folding_factor(0);
            let point =
                |seed: u64| Point::new((0..9).map(|c| EF::from_u64(seed + 3 * c)).collect());
            let points = [point(7), point(11)];
            let commit = |seed: u64, challenger: &mut Challenger| {
                let table = Table::rand(&mut SmallRng::seed_from_u64(seed), 1, 9);
                <Pcs<SuffixProver<F, EF>> as MultilinearPcs<EF, Challenger>>::commit(
                    &pcs,
                    SuffixProver::<F, EF>::new_witness(vec![table], folding),
                    challenger,
                )
                .unwrap()
                .1
            };
            let open = |device: bool| {
                let mut prover = challenger();
                let first = commit(1, &mut prover);
                let second = commit(2, &mut prover);
                let proof: PairProof<F, EF, Mmcs> = if device {
                    pcs.open_pair_at_on(
                        first,
                        &second,
                        &protocol,
                        &points,
                        &HostDevices(&pcs),
                        &mut prover,
                    )
                } else {
                    pcs.open_pair_at(first, &second, &protocol, &points, &mut prover)
                }
                .unwrap();
                postcard::to_allocvec(&proof).unwrap()
            };
            assert_eq!(open(true), open(false));
        }
    }

    #[test]
    fn a_device_single_opening_is_the_host_one() {
        for pow_bits in [0, 4] {
            let spec = TableSpec::new(
                TableShape::new(9, 1),
                vec![OpeningBatch::new(vec![0], Vec::new())],
            );
            let protocol = OpeningProtocol::new(vec![spec]);
            let pcs = pcs::<SuffixProver<F, EF>>(9, pow_bits);
            let folding = pcs.round_folding_factor(0);
            let point = Point::new((0..9).map(|c| EF::from_u64(5 + 3 * c)).collect());
            let open = |device: bool| {
                let mut prover = challenger();
                let table = Table::rand(&mut SmallRng::seed_from_u64(3), 1, 9);
                let data = <Pcs<SuffixProver<F, EF>> as MultilinearPcs<EF, Challenger>>::commit(
                    &pcs,
                    SuffixProver::<F, EF>::new_witness(vec![table], folding),
                    &mut prover,
                )
                .unwrap()
                .1;
                let points = core::slice::from_ref(&point);
                let proof =
                    if device {
                        pcs.open_at_on(data, &protocol, points, &HostDevices(&pcs), &mut prover)
                    } else {
                        <Pcs<SuffixProver<F, EF>> as p3_sumcheck::PrescribedPointPcs<
                            EF,
                            Challenger,
                        >>::open_at(
                            &pcs, data, &protocol, points, &mut prover
                        )
                    }
                    .unwrap();
                postcard::to_allocvec(&proof).unwrap()
            };
            assert_eq!(open(true), open(false));
        }
    }
}
