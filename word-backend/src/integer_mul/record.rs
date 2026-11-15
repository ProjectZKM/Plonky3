//! The transcript record of one multiplication reduction.

use alloc::vec::Vec;

use p3_sumcheck::generic_degree::GenericDegreeProof;
use serde::{Deserialize, Serialize};

/// Transcript record of one full-width unsigned multiplication reduction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntegerMulProof<F, EF> {
    /// Shared value of both exponent lifts at the sampled row point.
    pub(crate) root: EF,
    /// Reduction of the power tree, `(g^a)^b`, to its per-bit leaf evaluations.
    pub(crate) power: TreeProof<F, EF>,
    /// Sumcheck tying the twisted power leaves to the right factor and the base lift.
    pub(crate) selector: GenericDegreeProof<F, EF>,
    /// The base lift `g^a`, then every bit of the right factor, where the selector ends.
    pub(crate) selected: Vec<EF>,
    /// Reduction of the base tree, `g^a`, to the bits of the left factor.
    pub(crate) base: TreeProof<F, EF>,
    /// Reduction of the result tree, `g^(lo + 2^w * hi)`, to the bits of both limbs.
    pub(crate) result: TreeProof<F, EF>,
}

/// Transcript record of one product tree, root layer first.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreeProof<F, EF> {
    /// One reduction per layer, from the root towards the leaves.
    pub(crate) layers: Vec<LayerProof<F, EF>>,
    /// The row extension of every leaf column at the point the layers end on.
    pub(crate) leaves: Vec<EF>,
}

/// Transcript record of one product layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerProof<F, EF> {
    /// Sumcheck reducing one layer claim to a claim on the two halves below it.
    pub(crate) sumcheck: GenericDegreeProof<F, EF>,
    /// Evaluations of the lower layer with its new last coordinate at zero, then at one.
    pub(crate) halves: [EF; 2],
}
