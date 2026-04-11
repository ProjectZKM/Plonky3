//! Adapter implementing the multilinear PCS trait for the WHIR protocol.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, MultilinearOpenedValues, MultilinearPcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
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

        // ── Multi-column batching ────────────────────────────────────
        //
        // For width > 1, we batch N polynomials f_0, ..., f_{N-1} into
        // a single polynomial g = Σ α^i · f_i using a random challenge
        // α from the challenger. WHIR commits to g; the per-column
        // evaluations f_i(r) are computed and stored separately.
        //
        // For width == 1, we skip the batching step (no overhead).

        let eval_values = evaluations.values;
        let combined_evals = if width == 1 {
            eval_values.clone()
        } else {
            // Sample batching challenge from the Fiat-Shamir transcript.
            let alpha: EF = challenger.sample_algebra_element();

            // Compute g(x) = f_0(x) + α·f_1(x) + ... + α^{N-1}·f_{N-1}(x)
            // as an EF-valued polynomial, then project to F.
            // Since each f_i is F-valued, g is also F-valued when α ∈ F.
            // For α ∈ EF, we need to store g as EF-valued — but WHIR's
            // InitialStatement takes Poly<F>. So we batch in F by sampling
            // α from the base field via a hash of the extension element.
            //
            // Simplified approach: fold columns in base field using
            // powers of a base-field challenge derived from the transcript.
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
            // Multi-column: the combined polynomial's evaluation is
            // g(r) = Σ α^i · f_i(r). The individual f_i(r) are computed
            // from the original data and stored as opened values.
            // The WHIR proof validates g(r); the verifier checks the
            // linear combination.

            // Evaluate g at each point via the statement (registers the claim).
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

        // Absorb the protocol configuration into the Fiat-Shamir transcript.
        let ds = self.build_domain_separator();
        ds.observe_domain_separator(challenger);

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
        // Re-derive the same domain separator that the prover used, so
        // the verifier's transcript state is identical.
        let ds: DomainSeparator<EF, F> = self.build_domain_separator();
        ds.observe_domain_separator(challenger);

        // For multi-column: the verifier must first reconstruct the
        // batching challenge and compute the combined evaluation.
        // For single-column: use the claim directly.
        let combined_claims = if opening_claims.len() == 1 {
            opening_claims[0].clone()
        } else {
            // Sample the same batching challenge the prover used.
            let _alpha_ef: EF = challenger.sample_algebra_element();
            let alpha_base: F = challenger.sample_algebra_element();

            // Reconstruct g(r) = Σ α^i · f_i(r) from per-column claims.
            // All columns share the same opening points.
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

        // Parse the Merkle root and OOD answers from the proof.
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
