//! Errors returned by the multiplication reduction.

use p3_sumcheck::generic_degree::GenericDegreeError;
use thiserror::Error;

/// A challenge field too small for the lift, or a failed multiplication reduction.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum IntegerMulError {
    /// The multiplicative group is too small to hold a full-width product without collisions.
    #[error("a challenge field of order below 2^{required} cannot lift {required}-bit products")]
    FieldTooSmall {
        /// Bits of the widest product, twice the word width.
        required: usize,
    },
    /// The challenge field's order is not a power of two, so it has no Frobenius twist.
    #[error("the multiplication reduction needs a challenge field of characteristic two")]
    FieldNotBinary,
    /// The record carries a different number of values than one step sends.
    #[error("the {step} step carries {actual} values, expected {expected}")]
    LeafCount {
        /// Name of the step whose values differ.
        step: &'static str,
        /// Values fixed by the word width.
        expected: usize,
        /// Values the record carries.
        actual: usize,
    },
    /// The record carries a different number of layers than the tree depth.
    #[error("the {tree} tree carries {actual} layers, expected {expected}")]
    TreeDepth {
        /// Name of the tree whose depth differs.
        tree: &'static str,
        /// Depth fixed by the word width.
        expected: usize,
        /// Layers the record carries.
        actual: usize,
    },
    /// One delegated sumcheck is malformed or inconsistent.
    #[error("multiplication sumcheck failed: {0}")]
    Sumcheck(#[from] GenericDegreeError),
    /// A sumcheck claims a sum other than the claim the previous step left.
    #[error("a multiplication sumcheck does not start from the claim entering it")]
    EnteringClaim,
    /// A layer sumcheck does not close against the two halves the prover sent.
    #[error("a product layer does not close against its two halves")]
    LayerClaim,
    /// A product tree does not close against the leaf evaluations the prover sent.
    #[error("a product tree does not close against its leaf evaluations")]
    LeafClaim,
    /// The selector sumcheck does not close against the base lift and right bits it ends on.
    #[error("the selector does not close against the base lift and right bits")]
    SelectorClaim,
}
