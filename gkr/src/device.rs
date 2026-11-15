//! Product trees held by a device, driven layer by layer from the roots down.
//!
//! A caller that keeps its trees on a device builds [`ProductTrees`] from one. The layer prover
//! then reads the roots and the children a binary layer needs, and runs each radix-four layer's
//! sumcheck rounds there: the degree-five messages, the folds of the four slot columns and of
//! the equality weights, and the children it ends on. The messages are the ones the host
//! layers send, so the proof and the transcript are unchanged.

use alloc::boxed::Box;
use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};

use super::ROUND_POLY_LEN;
use super::proof::PreparedProductGkr;
use super::prover::LayerRounds;
use super::shape::ProductGkrShape;

/// Product trees on a device.
pub trait DeviceProductTrees<EF>: Send {
    /// Each tree's root.
    fn roots(&self) -> Vec<EF>;

    /// The first two values of each tree's layer at `depth`.
    fn binary_children(&self, depth: usize) -> Vec<[EF; 2]>;

    /// Start the radix-four layer whose children are each tree's layer at `depth`, at the
    /// point reached so far: row `i` of four children weighs the equality of the point with
    /// `i`, coordinate `k` binding bit `k` of `i`.
    fn begin_layer(&mut self, depth: usize, point: &[EF]);

    /// The current round's message: the batched product sum at the five interpolation
    /// `nodes`, the trees combined by the powers of `batching`.
    fn round(&self, nodes: &[EF; ROUND_POLY_LEN], batching: EF) -> [EF; ROUND_POLY_LEN];

    /// Bind the current round's variable to `challenge`.
    fn fold(&mut self, challenge: EF);

    /// Each tree's four children once every round of the layer is folded.
    fn children(&self) -> Vec<[EF; 4]>;
}

/// Separate product trees, their levels in host memory or on a device.
///
/// Their roots are known as soon as the levels are, so a caller can check a statement about
/// them before a transcript moves.
pub struct ProductTrees<EF: Clone + 'static> {
    pub(crate) trees: Trees<EF>,
}

pub(crate) enum Trees<EF: Clone + 'static> {
    Host(PreparedProductGkr<'static, EF>),
    Device(Box<dyn DeviceProductTrees<EF>>),
}

impl<EF: Field> ProductTrees<EF> {
    /// The trees over two owned leaf tables, built in host memory.
    ///
    /// # Panics
    ///
    /// Panics when the shape is batched or a table exceeds its logical tree.
    pub fn host<F: Field>(leaves: [Vec<EF>; 2], shape: ProductGkrShape) -> Self
    where
        EF: ExtensionField<F>,
    {
        Self {
            trees: Trees::Host(PreparedProductGkr::new_owned::<F>(leaves, shape)),
        }
    }

    /// Trees a device holds.
    #[must_use]
    pub fn from_device(device: Box<dyn DeviceProductTrees<EF>>) -> Self {
        Self {
            trees: Trees::Device(device),
        }
    }

    /// Each tree's product, in tree order.
    #[must_use]
    pub fn roots(&self) -> Vec<EF> {
        match &self.trees {
            Trees::Host(prepared) => prepared.roots().to_vec(),
            Trees::Device(device) => device.roots(),
        }
    }
}

/// One radix-four layer's rounds on the device, which tracks its own equality weights.
pub(crate) struct DeviceLayer<'a, EF>(pub(crate) &'a mut dyn DeviceProductTrees<EF>);

impl<EF: Field> LayerRounds<EF, 4> for DeviceLayer<'_, EF> {
    const READS_EQUALITY: bool = false;

    fn round(
        &mut self,
        _equality: &[EF],
        _equality_pair: [EF; 2],
        _logical_len: usize,
        batching: EF,
    ) -> [EF; ROUND_POLY_LEN] {
        let nodes = [0, 2, 3, 4, 5].map(EF::interpolation_node);
        self.0.round(&nodes, batching)
    }

    fn fold(&mut self, challenge: EF, _logical_len: usize) {
        self.0.fold(challenge);
    }

    fn children(&self) -> Vec<[EF; 4]> {
        self.0.children()
    }
}
