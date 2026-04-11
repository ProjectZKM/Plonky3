//! Adapter implementing the multilinear PCS trait for the WHIR protocol.
//!
//! # Protocol Overview (WHIR, Construction 5.1, ePrint 2024/1586)
//!
//! WHIR is an IOP of proximity for Reed-Solomon codes that achieves
//! super-fast verification. Given a multilinear polynomial
//! f: {0,1}^m → F with 2^m evaluations, the protocol proves that
//! f evaluates to claimed values at specified points.
//!
//! ## Single-column protocol
//!
//! 1. **Commit**: Encode f as a Reed-Solomon codeword via DFT.
//!    Commit the codeword rows in a Merkle tree.
//!    Register evaluation claims f(z_i) = v_i as equality constraints:
//!      sum_{b ∈ {0,1}^m} f(b) · eq(z_i, b) = v_i
//!    where eq(a, b) = ∏_j (a_j · b_j + (1 - a_j)(1 - b_j))
//!
//! 2. **Prove**: For each round i = 0..M-1:
//!    a. Run k_i sumcheck rounds to reduce the constraint claim
//!    b. Fold: f_{i+1}(X) = f_i(α, X) where α is a Fiat-Shamir challenge
//!    c. Commit the folded RS codeword
//!    d. Sample OOD points, verify consistency, grind for PoW
//!    e. Open Merkle paths at STIR query positions
//!    Final round: send polynomial coefficients in the clear.
//!
//! 3. **Verify**: Replay Fiat-Shamir, check sumcheck rounds, verify
//!    Merkle proofs, check final polynomial evaluation.
//!
//! ## Multi-column batch extension
//!
//! For N polynomials f_0, ..., f_{N-1} (columns of a trace matrix):
//!
//! 1. Absorb opening points into the Fiat-Shamir transcript
//! 2. Sample batching challenge α ∈ F
//! 3. Compute combined polynomial:
//!      g(x) = Σ_{j=0}^{N-1} α^j · f_j(x)
//! 4. Run single-column WHIR on g
//! 5. Provide per-column evaluations f_j(r) as opened values
//! 6. Verifier checks: g(r) = Σ α^j · f_j(r)
//!
//! Security: α must be sampled AFTER opening points are absorbed
//! (Fiat-Shamir binding). Column width N must also be absorbed.
//!
//! ## Soundness (Theorem 5.2)
//!
//! ε_WHIR ≤ ε_RS-proximity + ε_sumcheck + ε_OOD + 2^(-pow_bits)
//!
//! Under Johnson Bound (JBR) with extension degree D=5:
//!   ε_RS-proximity ≤ (1 - δ_JBR)^num_queries per round
//!   ε_sumcheck = d · N / |F^D| per round (negligible)
//!   ε_OOD = num_ood_samples / |F^D| per round (negligible)

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, MultilinearOpenedValues, MultilinearPcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{BasedVectorSpace, ExtensionField, Field, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use super::committer::reader::CommitmentReader;
use super::committer::writer::CommitmentWriter;
use super::proof::WhirProof;
use super::prover::Prover;
use super::verifier::Verifier;
use super::verifier::errors::VerifierError;
use crate::constraints::statement::EqStatement;
use crate::constraints::statement::initial::InitialStatement;
use crate::fiat_shamir::domain_separator::DomainSeparator;
use crate::parameters::{ProtocolParameters, SumcheckStrategy, WhirConfig};

/// WHIR-based multilinear polynomial commitment scheme.
///
/// Wraps the full WHIR IOP of proximity (Construction 5.1 in the paper)
/// behind a generic PCS trait.
///
/// The DFT backend and Fiat-Shamir domain separator are managed internally.
///
/// The const generic `DIGEST_ELEMS` must match the Merkle tree digest width
/// used by the underlying commitment scheme.
#[derive(Debug)]
pub struct WhirPcs<EF, F, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Full protocol configuration derived from the parameters.
    config: WhirConfig<EF, F, MT, Challenger>,
    /// Raw parameters kept around to allocate proof structures.
    protocol_params: ProtocolParameters<MT>,
    /// DFT backend for Reed-Solomon encoding (hidden from the trait surface).
    dft: Dft,
    /// Sumcheck proving strategy: classic constraint batching or split-value optimization.
    sumcheck_strategy: SumcheckStrategy,
}

/// Prover-side data produced by commit, consumed by open.
pub struct WhirProverData<F, EF, MT, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Merkle tree produced during commitment; used to open query positions.
    merkle_data: MT::ProverData<DenseMatrix<F>>,
    /// Statement carrying the polynomial and all equality constraints
    /// (both user-supplied evaluation claims and OOD challenge points).
    statement: InitialStatement<F, EF>,
    /// Proof structure with the initial commitment and OOD answers filled in.
    /// The proving phase fills the remaining round data.
    proof: WhirProof<F, EF, MT>,
    /// Evaluation values computed during commit, indexed per polynomial.
    opened_values: MultilinearOpenedValues<EF>,
}

impl<EF, F, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Create a new WHIR PCS for multilinear polynomials in `num_variables` variables.
    ///
    /// # Arguments
    ///
    /// - `num_variables`: dimension m (the polynomial has 2^m evaluations).
    /// - `protocol_params`: security level, folding factor, rate, Merkle tree, etc.
    /// - `dft`: the DFT backend used for Reed-Solomon encoding.
    /// - `sumcheck_strategy`: classic or split-value optimization.
    pub fn new(
        num_variables: usize,
        protocol_params: ProtocolParameters<MT>,
        dft: Dft,
        sumcheck_strategy: SumcheckStrategy,
    ) -> Self {
        // Derive the full round-by-round configuration from the raw parameters.
        let config = WhirConfig::new(num_variables, protocol_params.clone());
        Self {
            config,
            protocol_params,
            dft,
            sumcheck_strategy,
        }
    }

    /// Build the Fiat-Shamir domain separator for this protocol instance.
    ///
    /// The domain separator encodes all public protocol parameters into
    /// the transcript so the verifier's challenges are bound to this
    /// specific configuration (see Construction 5.1, step 1).
    fn build_domain_separator(&self) -> DomainSeparator<EF, F>
    where
        EF: TwoAdicField,
    {
        // Start with an empty pattern.
        let mut ds = DomainSeparator::new(vec![]);
        // Encode the public parameters (num_variables, security, rate, etc.).
        ds.commit_statement::<MT, Challenger, DIGEST_ELEMS>(&self.config);
        // Encode the full proof structure (round counts, query counts, etc.).
        ds.add_whir_proof::<MT, Challenger, DIGEST_ELEMS>(&self.config);
        ds
    }
}

impl<EF, F, MT, Challenger, Dft, const DIGEST_ELEMS: usize> MultilinearPcs<EF, Challenger>
    for WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    type Val = F;
    type Commitment = MT::Commitment;
    type ProverData = WhirProverData<F, EF, MT, DIGEST_ELEMS>;
    type Proof = WhirProof<F, EF, MT>;
    type Error = VerifierError;

    fn num_vars(&self) -> usize {
        self.config.num_variables
    }

    fn commit(
        &self,
        evaluations: RowMajorMatrix<Self::Val>,
        opening_points: &[Vec<Point<EF>>],
        challenger: &mut Challenger,
    ) -> (Self::Commitment, Self::ProverData) {
        let n = 1 << self.config.num_variables;
        let width = evaluations.width();
        assert_eq!(
            evaluations.height(),
            n,
            "evaluation vector length must be 2^num_variables"
        );

        // ── Fiat-Shamir binding ────────────────────────────────────
        //
        // SECURITY: Absorb opening points into the transcript BEFORE
        // sampling any challenges. This prevents a malicious prover from
        // choosing opening points after seeing the batching challenge.

        // Absorb domain separator first (binds protocol parameters).
        let ds = self.build_domain_separator();
        ds.observe_domain_separator(challenger);

        // Absorb all opening points into the transcript.
        // Points are EF-valued; decompose into base field elements for the challenger.
        for point_set in opening_points {
            for point in point_set {
                for &coord in point.as_slice() {
                    challenger.observe_algebra_element(coord);
                }
            }
        }

        // Absorb the number of columns (binds the batching structure).
        // Absorb width as field elements to bind batching structure.
        for _ in 0..width {
            challenger.observe(F::ONE);
        }

        // ── Multi-column batching ────────────────────────────────────
        //
        // For width > 1, we batch N polynomials f_0, ..., f_{N-1} into
        // a single polynomial g = Σ α^i · f_i using a random challenge
        // α sampled AFTER opening points are bound to the transcript.

        let eval_values = evaluations.values;
        let combined_evals = if width == 1 {
            eval_values.clone()
        } else {
            // Sample batching challenge α ∈ F from the transcript.
            // Sampled AFTER opening points are absorbed (Fiat-Shamir binding).
            let alpha_base: F = challenger.sample_algebra_element();
            let mut combined = vec![F::ZERO; n];
            let mut alpha_pow = F::ONE;
            let vals = &eval_values;
            for col in 0..width {
                for row in 0..n {
                    combined[row] += alpha_pow * vals[row * width + col];
                }
                alpha_pow *= alpha_base;
            }
            combined
        };

        // Wrap the combined evaluation vector as a multilinear polynomial.
        let poly = Poly::new(combined_evals);

        // Build the initial statement and register evaluation claims.
        let mut statement = self.config.initial_statement(poly, self.sumcheck_strategy);

        // For each polynomial (column), evaluate at each opening point.
        let mut all_values = Vec::with_capacity(opening_points.len().max(1));

        if width == 1 {
            // Single column: evaluate directly.
            assert!(opening_points.len() <= 1);
            if !opening_points.is_empty() {
                let mut col_values = Vec::with_capacity(opening_points[0].len());
                for point in &opening_points[0] {
                    let eval = statement.evaluate(point);
                    col_values.push(eval);
                }
                all_values.push(col_values);
            }
        } else {
            // Multi-column: evaluate g at each point (registers the claim).
            if !opening_points.is_empty() {
                for point in &opening_points[0] {
                    let _combined_eval = statement.evaluate(point);
                }
            }

            // Compute per-column evaluations at each opening point.
            for col in 0..width {
                let col_evals: Vec<F> = (0..n)
                    .map(|row| eval_values[row * width + col])
                    .collect();
                let col_poly = Poly::new(col_evals);

                let mut col_values = Vec::new();
                if !opening_points.is_empty() {
                    for point in &opening_points[0] {
                        col_values.push(col_poly.eval_base(point));
                    }
                }
                all_values.push(col_values);
            }
        }

        // Allocate the proof structure with pre-sized vectors for each round.
        let mut proof =
            WhirProof::from_protocol_parameters(&self.protocol_params, self.config.num_variables);

        // Run the commitment phase.
        let committer = CommitmentWriter::new(&self.config);
        let merkle_data = committer
            .commit(&self.dft, &mut proof, challenger, &mut statement)
            .expect("commitment phase failed");

        // The Merkle root is now stored in the proof.
        let commitment = proof
            .initial_commitment
            .clone()
            .expect("commitment should be set after commit phase");

        // Bundle everything the prover needs for the opening phase.
        let prover_data = WhirProverData {
            merkle_data,
            statement,
            proof,
            opened_values: all_values,
        };

        (commitment, prover_data)
    }

    fn open(
        &self,
        mut prover_data: Self::ProverData,
        challenger: &mut Challenger,
    ) -> (MultilinearOpenedValues<EF>, Self::Proof) {
        // Execute the multi-round WHIR proving protocol (Construction 5.1):
        //   For each round i = 0..M-1:
        //     1. Run k_i sumcheck rounds to reduce the constraint claim.
        //     2. Fold the polynomial: f_{i+1}(X) = f_i(alpha, X).
        //     3. Commit the folded codeword via a Merkle tree.
        //     4. Sample OOD points and verify consistency.
        //     5. Perform proof-of-work grinding to bind the transcript.
        //     6. Generate STIR query positions and open Merkle paths.
        //   Final round: send the polynomial coefficients in the clear.
        let prover = Prover(&self.config);
        prover
            .prove(
                &self.dft,
                &mut prover_data.proof,
                challenger,
                &prover_data.statement,
                prover_data.merkle_data,
            )
            .expect("proving phase failed");

        (prover_data.opened_values, prover_data.proof)
    }

    fn verify(
        &self,
        _commitment: &Self::Commitment,
        opening_claims: &[Vec<(Point<EF>, EF)>],
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        // ── Fiat-Shamir binding (must match prover order exactly) ──
        //
        // SECURITY: Domain separator → opening points → width → α → proof.

        // 1. Domain separator (binds protocol parameters).
        let ds: DomainSeparator<EF, F> = self.build_domain_separator();
        ds.observe_domain_separator(challenger);

        // 2. Absorb opening points from claims (same as prover).
        for col_claims in opening_claims {
            for (point, _) in col_claims {
                for &coord in point.as_slice() {
                    challenger.observe_algebra_element(coord);
                }
            }
        }

        // 3. Absorb width (number of columns).
        let width = opening_claims.len();
        // Absorb width as field elements to bind batching structure.
        for _ in 0..width {
            challenger.observe(F::ONE);
        }

        // 4. Reconstruct batching challenge and combined claims.
        let combined_claims = if width == 1 {
            opening_claims[0].clone()
        } else {
            // Sample the same batching challenge α the prover used.
            let alpha_base: F = challenger.sample_algebra_element();

            // Reconstruct g(r) = Σ α^i · f_i(r) from per-column claims.
            let num_points = opening_claims[0].len();
            let mut combined = Vec::with_capacity(num_points);
            for pt_idx in 0..num_points {
                let point = opening_claims[0][pt_idx].0.clone();
                let mut combined_val = EF::ZERO;
                let mut alpha_pow = EF::ONE;
                let alpha_ef: EF = alpha_base.into();
                for col_claims in opening_claims {
                    combined_val += alpha_pow * col_claims[pt_idx].1;
                    alpha_pow *= alpha_ef;
                }
                combined.push((point, combined_val));
            }
            combined
        };

        // 5. Parse the Merkle root and OOD answers from the proof.
        let commitment_reader = CommitmentReader::new(&self.config);
        let parsed_commitment =
            commitment_reader.parse_commitment::<F, DIGEST_ELEMS>(proof, challenger);

        // Reconstruct the equality statement from the combined claims.
        let mut eq_statement = EqStatement::initialize(self.config.num_variables);
        for (point, value) in &combined_claims {
            eq_statement.add_evaluated_constraint(point.clone(), *value);
        }

        // Run the full WHIR verification.
        let verifier = Verifier::new(&self.config);
        verifier.verify(proof, challenger, &parsed_commitment, eq_statement)?;

        Ok(())
    }
}
