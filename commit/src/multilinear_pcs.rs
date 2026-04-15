//! Polynomial commitment scheme trait for multilinear polynomials.

use alloc::vec::Vec;
use core::fmt::Debug;

use p3_field::ExtensionField;
use p3_matrix::dense::RowMajorMatrix;
use serde::de::DeserializeOwned;
use serde::Serialize;

/// A point in the multilinear evaluation domain.
/// Re-exported from p3-multilinear-util when available.
pub type MultilinearPoint<F> = Vec<F>;

/// Claimed evaluation values from an opening.
///
/// `values[i][j]` = evaluation of polynomial i at its j-th opening point.
pub type MultilinearOpenedValues<F> = Vec<Vec<F>>;

/// Polynomial commitment scheme for multilinear polynomials over the Boolean hypercube.
pub trait MultilinearPcs<Challenge, Challenger>
where
    Challenge: ExtensionField<Self::Val>,
{
    /// Base field of the committed polynomials.
    type Val: p3_field::Field;

    /// Succinct binding commitment sent to the verifier.
    type Commitment: Clone + Serialize + DeserializeOwned;

    /// Prover-side auxiliary data retained between commit and open.
    type ProverData;

    /// Opening proof checked by the verifier.
    type Proof: Clone + Serialize + DeserializeOwned;

    /// Verification failure type.
    type Error: Debug;

    /// Number of variables m of the committed polynomials.
    fn num_vars(&self) -> usize;

    /// Commit to a batch of multilinear polynomials and register opening points.
    fn commit(
        &self,
        evaluations: RowMajorMatrix<Self::Val>,
        opening_points: &[Vec<MultilinearPoint<Challenge>>],
        challenger: &mut Challenger,
    ) -> (Self::Commitment, Self::ProverData);

    /// Produce an opening proof for the points registered during commit.
    fn open(
        &self,
        prover_data: Self::ProverData,
        challenger: &mut Challenger,
    ) -> (MultilinearOpenedValues<Challenge>, Self::Proof);

    /// Verify an opening proof against a commitment and claimed evaluations.
    fn verify(
        &self,
        commitment: &Self::Commitment,
        opening_claims: &[Vec<(MultilinearPoint<Challenge>, Challenge)>],
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error>;
}
