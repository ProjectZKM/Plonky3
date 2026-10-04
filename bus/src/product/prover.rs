//! Identity-padded storage and folding state used by the product prover.
//!
//! Every pass over a level runs in parallel. A level is held once: the leaves are taken by
//! value, and a radix-four layer reads its level in place until its first fold halves it.

use alloc::vec::Vec;

use p3_field::Field;
use p3_maybe_rayon::prelude::*;

use super::ROUND_POLY_LEN;
use super::math::interpolate_pair;

/// Rows one task of a sumcheck round sums before its message joins the others.
const ROUND_CHUNK: usize = 1 << 12;

/// An arbitrary table prefix whose omitted suffix is the multiplicative identity.
struct IdentityPrefix<F> {
    /// Explicit values after removing trailing identities.
    values: Vec<F>,
}

impl<F: Field> IdentityPrefix<F> {
    /// Creates a canonical explicit prefix from a complete or partial table.
    fn new(mut values: Vec<F>) -> Self {
        // Trailing identities have the same semantics when left implicit.
        while values.last() == Some(&F::ONE) {
            values.pop();
        }
        Self { values }
    }

    /// Reads one logical value through implicit identity padding.
    #[inline]
    fn get(&self, index: usize) -> F {
        self.values.get(index).copied().unwrap_or(F::ONE)
    }

    /// Multiplies fixed-size groups into the next retained product layer.
    fn reduce(&self, arity: usize) -> Self {
        // A partial final group is completed by implicit identity factors.
        let values = self
            .values
            .par_chunks(arity)
            .map(|chunk| chunk.iter().copied().product())
            .collect();
        Self::new(values)
    }
}

/// Product levels retained for one identity-padded input prefix.
pub(super) struct ProductLayers<F> {
    /// Explicit prefixes indexed by their base-two depth above the leaves.
    layers: Vec<IdentityPrefix<F>>,
}

impl<F: Field> ProductLayers<F> {
    /// Builds every product level needed by the radix-four descent, keeping `leaves` as the
    /// lowest one.
    pub(super) fn new(leaves: Vec<F>, log_height: usize) -> Self {
        // Unvisited depths remain empty constant-one prefixes.
        let mut layers = (0..=log_height)
            .map(|_| IdentityPrefix::new(Vec::new()))
            .collect::<Vec<_>>();
        layers[0] = IdentityPrefix::new(leaves);

        // Two multiplication levels are retained per radix-four layer.
        let mut depth = 0;
        while depth + 2 <= log_height {
            layers[depth + 2] = layers[depth].reduce(4);
            depth += 2;
        }
        if depth < log_height {
            layers[log_height] = layers[depth].reduce(2);
        }

        Self { layers }
    }

    /// Returns the fully reduced product root.
    #[inline]
    pub(super) fn root(&self) -> F {
        self.layers
            .last()
            .expect("every product tree retains its root layer")
            .get(0)
    }

    /// Reads the two children of a root-most binary layer.
    #[inline]
    pub(super) fn binary_children(&self, depth: usize) -> [F; 2] {
        [self.layers[depth].get(0), self.layers[depth].get(1)]
    }
}

/// Four child multilinears represented as arbitrary prefixes of constant-one tables.
///
/// Child `slot` holds the level's entries `4 j + slot`. Before the first fold they are read
/// where the level holds them; the fold writes each child out at half the length.
enum RadixFourState<'a, F> {
    /// The retained level, its four children interleaved.
    Level(&'a IdentityPrefix<F>),
    /// The children after a fold, one prefix each.
    Folded([IdentityPrefix<F>; 4]),
}

impl<F: Field> RadixFourState<'_, F> {
    /// Entry `row` of child `slot`, through implicit identity padding.
    #[inline]
    fn get(&self, slot: usize, row: usize) -> F {
        match self {
            Self::Level(level) => level.get(4 * row + slot),
            Self::Folded(children) => children[slot].get(row),
        }
    }

    /// Explicit entries child `slot` holds.
    const fn explicit_len(&self, slot: usize) -> usize {
        match self {
            Self::Level(level) => level.values.len().saturating_sub(slot).div_ceil(4),
            Self::Folded(children) => children[slot].values.len(),
        }
    }

    /// Binds one parent variable in every child multilinear.
    fn fold(&mut self, challenge: F) {
        let children = core::array::from_fn(|slot| {
            // Missing entries retain the constant-one suffix during interpolation.
            let values = (0..self.explicit_len(slot).div_ceil(2))
                .into_par_iter()
                .map(|row| {
                    interpolate_pair(
                        [self.get(slot, 2 * row), self.get(slot, 2 * row + 1)],
                        challenge,
                    )
                })
                .collect();
            IdentityPrefix::new(values)
        });
        *self = Self::Folded(children);
    }

    /// Reads the four terminal child claims after every parent variable is bound.
    fn children(&self) -> [F; 4] {
        core::array::from_fn(|slot| self.get(slot, 0))
    }
}

/// Per-tree states reduced by one shared radix-four sumcheck.
pub(super) struct RadixFourBatch<'a, F> {
    /// One folding state for each product tree.
    states: Vec<RadixFourState<'a, F>>,
}

impl<'a, F: Field> RadixFourBatch<'a, F> {
    /// Creates the batched states over one retained level per tree, read in place.
    pub(super) fn new(layers: &'a [ProductLayers<F>], depth: usize) -> Self {
        let states = layers
            .iter()
            .map(|layers| RadixFourState::Level(&layers.layers[depth]))
            .collect();
        Self { states }
    }

    /// Computes one degree-five batched sumcheck message.
    pub(super) fn round(
        &self,
        equality: &[F],
        logical_len: usize,
        batching: F,
    ) -> [F; ROUND_POLY_LEN] {
        debug_assert!(logical_len >= 2);
        debug_assert_eq!(equality.len(), logical_len);

        // Node one is omitted because the running sum reconstructs it.
        let nodes = [0, 2, 3, 4, 5].map(F::interpolation_node);
        let rows = logical_len / 2;

        // Each chunk of rows sums its own message; the chunks' messages are added at the end.
        let partials = (0..rows.div_ceil(ROUND_CHUNK))
            .into_par_iter()
            .map(|chunk| {
                let mut evaluations = [F::ZERO; ROUND_POLY_LEN];
                for row in chunk * ROUND_CHUNK..((chunk + 1) * ROUND_CHUNK).min(rows) {
                    let eq_zero = equality[2 * row];
                    let eq_one = equality[2 * row + 1];
                    for (node_index, node) in nodes.into_iter().enumerate() {
                        let eq_value = interpolate_pair([eq_zero, eq_one], node);
                        let mut power = F::ONE;
                        let mut batched_product = F::ZERO;
                        for state in &self.states {
                            let product = (0..4)
                                .map(|slot| {
                                    let zero = state.get(slot, 2 * row);
                                    let one = state.get(slot, 2 * row + 1);
                                    interpolate_pair([zero, one], node)
                                })
                                .product::<F>();
                            batched_product += power * product;
                            power *= batching;
                        }
                        evaluations[node_index] += eq_value * batched_product;
                    }
                }
                evaluations
            })
            .collect::<Vec<_>>();
        partials
            .into_iter()
            .fold([F::ZERO; ROUND_POLY_LEN], |mut sum, partial| {
                for (sum, partial) in sum.iter_mut().zip(partial) {
                    *sum += partial;
                }
                sum
            })
    }

    /// Binds one parent variable across every tree in the batch.
    pub(super) fn fold(&mut self, challenge: F) {
        for state in &mut self.states {
            state.fold(challenge);
        }
    }

    /// Collects the terminal child claims in tree order.
    pub(super) fn children(&self) -> Vec<[F; 4]> {
        self.states.iter().map(RadixFourState::children).collect()
    }
}
