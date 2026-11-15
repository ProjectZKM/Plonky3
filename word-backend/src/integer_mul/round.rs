//! The cubic sumcheck provers of the multiplication reduction.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cell::Cell;

use p3_field::Field;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_sumcheck::generic_degree::RoundProver;

use crate::device::{DeviceSelector, DeviceTree, from_device, slice_to_device, to_device};

/// Per-variable degree of every sumcheck the multiplication reduction runs.
///
/// Each round is an equality factor times at most two tables varying in any one variable.
pub(super) const DEGREE: usize = 3;

/// The nodes every round polynomial is sent at; the value at one follows from the claim.
fn nodes<F: Field>() -> [F; DEGREE] {
    [F::ZERO, F::interpolation_node(2), F::interpolation_node(3)]
}

/// Binds the most significant variable of a table to `challenge`, in place.
fn fold_table<F: Field>(table: &mut Vec<F>, challenge: F) {
    let half = table.len() / 2;
    let (low, high) = table.split_at_mut(half);
    low.par_iter_mut()
        .zip(high.par_iter())
        .for_each(|(low, &high)| *low += challenge * (high - *low));
    table.truncate(half);
}

/// Sums `value(row, node)` over the lower half of the cube at every node.
fn round_sums<F: Field>(half: usize, value: impl Fn(usize, F) -> F + Sync) -> Vec<F> {
    let nodes = nodes::<F>();
    let sums = (0..half).into_par_iter().par_fold_reduce(
        || [F::ZERO; DEGREE],
        |mut acc, row| {
            for (slot, &node) in acc.iter_mut().zip(&nodes) {
                *slot += value(row, node);
            }
            acc
        },
        |mut acc, other| {
            for (slot, value) in acc.iter_mut().zip(other) {
                *slot += value;
            }
            acc
        },
    );
    sums.to_vec()
}

/// Prover state for one product layer, with the equality factor of the bound variable split off.
///
/// ```text
///     sum_z eq(r, z) * zero(z) * one(z)
///
///     h_i(X) = scale * eq(r_i, X) * q(X)
///     q(X)   = sum_rest eq(r_rest, rest) * zero(X, rest) * one(X, rest)
/// ```
///
/// - `q` is quadratic, so its value at zero and its leading coefficient fix it with the claim.
/// - The rest weights split into two half-width tables, so no full equality table is stored.
/// - On a device the tree holds the lower layer, and only the sums cross to the host.
pub(super) struct ProductLayer<'a, F> {
    /// The layer point, most significant variable first.
    point: Vec<F>,
    /// The lower layer at last coordinate zero, then at last coordinate one.
    tables: Tables<'a, F>,
    /// Variables bound so far.
    round: usize,
    /// Product of the equality factors of the bound variables.
    scale: F,
    /// The claim the current round splits.
    claim: F,
    /// `q(0)`, `q(1)`, and the leading coefficient of the last round sent.
    quadratic: Cell<[F; 3]>,
}

/// Where the lower layer of a product layer lives.
enum Tables<'a, F> {
    /// Both halves on the host.
    Host([Vec<F>; 2]),
    /// The tree on the device, its layer sumcheck begun.
    Device(&'a mut dyn DeviceTree),
}

impl<'a, F: Field> ProductLayer<'a, F> {
    /// Creates a prover state for a layer claim at `point`.
    pub(super) fn new(point: Vec<F>, tables: [Vec<F>; 2], claim: F) -> Self {
        // Both halves span the cube the point addresses.
        debug_assert!(tables.iter().all(|table| table.len() == 1 << point.len()));
        Self::with_tables(point, Tables::Host(tables), claim)
    }

    /// Creates a prover state for the claim on layer `layer` of a device tree at `point`.
    pub(super) fn on_device(
        point: Vec<F>,
        tree: &'a mut dyn DeviceTree,
        layer: usize,
        claim: F,
    ) -> Self {
        tree.begin(layer, &slice_to_device(&point));
        Self::with_tables(point, Tables::Device(tree), claim)
    }

    fn with_tables(point: Vec<F>, tables: Tables<'a, F>, claim: F) -> Self {
        Self {
            point,
            tables,
            round: 0,
            scale: F::ONE,
            claim,
            quadratic: Cell::new([F::ZERO; 3]),
        }
    }

    /// Returns both halves once all variables are bound.
    pub(super) fn terminal(&self) -> [F; 2] {
        match &self.tables {
            Tables::Host(tables) => tables.each_ref().map(|table| {
                debug_assert_eq!(table.len(), 1, "every sumcheck variable was bound");
                table[0]
            }),
            Tables::Device(tree) => tree.halves().map(from_device),
        }
    }

    /// Sums `weight(rest) * term(zero(0, rest), zero(1, rest), one(0, rest), one(1, rest))`.
    fn weighted<const N: usize>(
        &self,
        [zero, one]: &[Vec<F>; 2],
        term: impl Fn([F; 4]) -> [F; N] + Sync,
    ) -> [F; N] {
        let rest = &self.point[self.round + 1..];
        let (high, low) = rest.split_at(rest.len() / 2);
        let high = Point::new(high).equality_weights_msb();
        let low = Point::new(low).equality_weights_msb();
        let half = zero.len() / 2;

        // The low weights scale one contiguous run of rows; the high weight scales the run's sum.
        high.par_iter().enumerate().par_fold_reduce(
            || [F::ZERO; N],
            |mut acc, (block, &outer)| {
                let mut inner = [F::ZERO; N];
                for (offset, &weight) in low.iter().enumerate() {
                    let row = block * low.len() + offset;
                    let terms = term([zero[row], zero[row + half], one[row], one[row + half]]);
                    for (slot, value) in inner.iter_mut().zip(terms) {
                        *slot += weight * value;
                    }
                }
                for (slot, value) in acc.iter_mut().zip(inner) {
                    *slot += outer * value;
                }
                acc
            },
            |mut acc, other| {
                for (slot, value) in acc.iter_mut().zip(other) {
                    *slot += value;
                }
                acc
            },
        )
    }

    /// Whether the claim of a round at coordinate `r` fixes its sum at one: when both the scale
    /// and `r` invert.
    fn claim_fixes_one(&self, r: F) -> bool {
        !self.scale.is_zero() && !r.is_zero()
    }

    /// Evaluates the current round polynomial at `x` from the stored quadratic.
    fn round_at(&self, x: F) -> F {
        let r = self.point[self.round];
        let [at_zero, at_one, leading] = self.quadratic.get();
        let linear = at_one - at_zero - leading;
        let q = at_zero + (linear + leading * x) * x;
        self.scale * ((F::ONE - r) * (F::ONE - x) + r * x) * q
    }
}

impl<F: Field> RoundProver<F> for ProductLayer<'_, F> {
    fn fold(&mut self, challenge: F) {
        // The next claim is this round's polynomial at the challenge.
        self.claim = self.round_at(challenge);
        let r = self.point[self.round];
        self.scale *= (F::ONE - r) * (F::ONE - challenge) + r * challenge;
        let next_at_one = self
            .point
            .get(self.round + 1)
            .is_some_and(|&r| !self.claim_fixes_one(r));
        match &mut self.tables {
            Tables::Host(tables) => {
                for table in tables {
                    fold_table(table, challenge);
                }
            }
            Tables::Device(tree) => {
                tree.fold(to_device(challenge).expect("the device's field"), next_at_one);
            }
        }
        self.round += 1;
    }

    fn round_poly(&self) -> Vec<F> {
        let r = self.point[self.round];
        // The device returns the direct sum at one with the other two.
        let ([at_zero, leading], direct) = match &self.tables {
            Tables::Host(tables) => (
                self.weighted(tables, |[a0, a1, b0, b1]| [a0 * b0, (a1 - a0) * (b1 - b0)]),
                None,
            ),
            Tables::Device(tree) => {
                let [at_zero, leading, at_one] = tree.sums(!self.claim_fixes_one(r)).map(from_device);
                ([at_zero, leading], Some(at_one))
            }
        };

        // The claim is scale * ((1 - r) * q(0) + r * q(1)), which fixes q(1) when both invert.
        let at_one = match (self.scale.try_inverse(), r.try_inverse()) {
            (Some(scale), Some(r_inverse)) => {
                (self.claim * scale - (F::ONE - r) * at_zero) * r_inverse
            }
            _ => direct.unwrap_or_else(|| match &self.tables {
                Tables::Host(tables) => self.weighted(tables, |[_, a1, _, b1]| [a1 * b1])[0],
                Tables::Device(_) => unreachable!("the device sends the sum at one"),
            }),
        };
        self.quadratic.set([at_zero, at_one, leading]);
        nodes::<F>()
            .iter()
            .map(|&node| self.round_at(node))
            .collect()
    }
}

/// Prover state for the twisted selector sum of the power leaves.
///
/// ```text
///     sum_x sum_i weight_i(x) * (1 + bit_i(x) * (base(x) - 1))
///
///     weight_i(x) = eq(gamma, i) * eq(phi^(-i)(r), x)
/// ```
pub(super) enum Selector<F> {
    /// Every table on the host.
    Host {
        /// One weight table per bit.
        weights: Vec<Vec<F>>,
        /// One bit table of the right factor per bit.
        bits: Vec<Vec<F>>,
        /// The base lift `g^a` of every row.
        base: Vec<F>,
    },
    /// Every table on the device.
    Device(Box<dyn DeviceSelector>),
}

impl<F: Field> Selector<F> {
    /// Creates a prover state over tables of one shared length.
    pub(super) fn new(weights: Vec<Vec<F>>, bits: Vec<Vec<F>>, base: Vec<F>) -> Self {
        debug_assert_eq!(weights.len(), bits.len());
        Self::Host {
            weights,
            bits,
            base,
        }
    }

    /// The nodes the device sends every round at.
    pub(super) fn device_nodes() -> [u128; DEGREE] {
        nodes::<F>().map(|node| to_device(node).expect("the device's field"))
    }

    /// Returns the base lift, then every bit, once all variables are bound.
    pub(super) fn terminal(&self) -> Vec<F> {
        match self {
            Self::Host { bits, base, .. } => {
                let mut values = Vec::with_capacity(1 + bits.len());
                values.push(base[0]);
                values.extend(bits.iter().map(|bits| bits[0]));
                values
            }
            Self::Device(device) => device.terminal().into_iter().map(from_device).collect(),
        }
    }
}

impl<F: Field> RoundProver<F> for Selector<F> {
    fn fold(&mut self, challenge: F) {
        match self {
            Self::Host {
                weights,
                bits,
                base,
            } => {
                fold_table(base, challenge);
                for table in weights.iter_mut().chain(bits.iter_mut()) {
                    fold_table(table, challenge);
                }
            }
            Self::Device(device) => device.fold(to_device(challenge).expect("the device's field")),
        }
    }

    fn round_poly(&self) -> Vec<F> {
        match self {
            Self::Host {
                weights,
                bits,
                base,
            } => {
                let half = base.len() / 2;
                round_sums(half, |row, node| {
                    let at = |table: &[F]| table[row] + (table[row + half] - table[row]) * node;
                    let lift = at(base) - F::ONE;
                    weights
                        .iter()
                        .zip(bits)
                        .map(|(weights, bits)| at(weights) * (F::ONE + at(bits) * lift))
                        .sum()
                })
            }
            Self::Device(device) => device.round().into_iter().map(from_device).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_binary_field::{BinaryField128, TowerLevel};
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::integer_mul::exponent::evaluate;

    type F = BinaryField128;

    fn table(seed: u128, len: u128) -> Vec<F> {
        (0..len).map(|row| F::from_repr(seed * row + 3)).collect()
    }

    // Runs a three-variable layer and checks every round against the definition.
    fn rounds_match_the_definition(point: [F; 3]) {
        let (zero, one) = (table(2, 8), table(7, 8));
        let eq = Point::new(point.as_slice()).equality_weights_msb();
        let claim = (0..8).map(|z| eq[z] * zero[z] * one[z]).sum();
        let mut prover = ProductLayer::new(point.to_vec(), [zero.clone(), one.clone()], claim);

        // Invariant: each round, read at every node, equals the sum with that variable fixed.
        let mut bound = Vec::new();
        for (round, challenge) in [19, 23, 29].map(F::from_repr).into_iter().enumerate() {
            let rest = point.len() - round - 1;
            for (&node, value) in nodes::<F>().iter().zip(prover.round_poly()) {
                let direct = (0..1_usize << rest)
                    .map(|tail| {
                        let mut at = bound.clone();
                        at.push(node);
                        at.extend(
                            (0..rest)
                                .rev()
                                .map(|bit| F::from_bool((tail >> bit) & 1 == 1)),
                        );
                        Point::eval_eq(&point, &at) * evaluate(&zero, &at) * evaluate(&one, &at)
                    })
                    .sum::<F>();
                assert_eq!(value, direct, "round {round}");
            }
            prover.fold(challenge);
            bound.push(challenge);
        }

        // The halves end on each table's extension at the challenges.
        assert_eq!(
            prover.terminal(),
            [evaluate(&zero, &bound), evaluate(&one, &bound)]
        );
    }

    #[test]
    fn every_layer_round_matches_its_definition() {
        rounds_match_the_definition([5, 11, 17].map(F::from_repr));
    }

    #[test]
    fn a_zero_coordinate_takes_the_direct_sum() {
        // The value at one cannot follow from the claim when its coordinate is zero.
        rounds_match_the_definition([F::from_repr(5), F::ZERO, F::from_repr(17)]);
    }

    #[test]
    fn the_selector_round_matches_its_definition() {
        // Fixture state: two bits over a two-variable cube.
        let weights = vec![table(3, 4), table(7, 4)];
        let bits = vec![
            vec![F::ONE, F::ZERO, F::ONE, F::ONE],
            vec![F::ZERO, F::ONE, F::ONE, F::ZERO],
        ];
        let base = table(13, 4);
        let prover = Selector::new(weights.clone(), bits.clone(), base.clone());
        let value = |row: usize| {
            (0..2)
                .map(|i| weights[i][row] * (F::ONE + bits[i][row] * (base[row] - F::ONE)))
                .sum::<F>()
        };
        assert_eq!(prover.round_poly()[0], value(0) + value(1));
    }
}
