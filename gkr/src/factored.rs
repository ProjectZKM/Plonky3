//! Round messages with the bound coordinate's equality factor divided out.
//!
//! Every eq-weighted round polynomial is `eq(anchor, X) g(X)`, where `anchor` is the parent
//! point's coordinate the round binds and `g` carries the coordinates bound before it.
//! A factored round sends the coefficients of `g` above its constant term.
//! The verifier rebuilds the constant from the running sum, since
//! `s(0) + s(1) = (1 - anchor) g(0) + anchor g(1) = g(0) + anchor (g(1) - g(0))`,
//! and `g(1) - g(0)` is the sum of the sent coefficients.
//!
//! A radix-four round then sends four values where the degree-five format sends five.
//! A binary round, whose polynomial has degree three, sends two.
//! The verifier checks the same running sums as the degree-five format, with no inversion.

use alloc::vec::Vec;

use p3_field::Field;
use p3_sumcheck::generic_degree::RoundPolyInterpolator;

use crate::ROUND_POLY_LEN;

/// Coefficients one factored radix-four round sends.
pub const FACTORED_LEN: usize = 4;

/// Coefficients one factored binary round sends.
pub const FACTORED_BINARY_LEN: usize = 2;

/// The equality line `eq(anchor, x) = (1 - anchor)(1 - x) + anchor x`.
pub(crate) fn equality_line<F: Field>(anchor: F, x: F) -> F {
    (F::ONE - anchor) * (F::ONE - x) + anchor * x
}

/// The verifier's next running sum after a factored round.
///
/// `message` holds the coefficients of `g` above the constant, `claim` the running sum before
/// the round, `anchor` the coordinate it binds and `challenge` the value bound to it.
pub(crate) fn factored_step<F: Field>(message: &[F], claim: F, anchor: F, challenge: F) -> F {
    let constant = claim - anchor * message.iter().copied().sum::<F>();
    let mut value = F::ZERO;
    for &coefficient in message.iter().rev() {
        value = (value + coefficient) * challenge;
    }
    equality_line(anchor, challenge) * (value + constant)
}

/// Rewrites one layer's degree-five round messages as the factored ones.
///
/// The prover tracks the running sum, so the omitted node one is known.
/// The cofactor is the message divided by the equality line wherever the line is nonzero,
/// which fails at one node at most, so six nodes always leave enough points.
pub(crate) struct Factoring<F> {
    /// Parent-point coordinates in binding order.
    anchors: Vec<F>,
    /// Running sum before the current round.
    claim: F,
    /// Rounds already bound.
    round: usize,
    /// Degree of the cofactor: four for a radix-four layer, two for a binary one.
    degree: usize,
    /// Evaluates a degree-five message at its challenge.
    interpolator: RoundPolyInterpolator<F>,
}

impl<F: Field> Factoring<F> {
    /// The rewriter of a layer over `anchors` whose running sum starts at `claim`.
    pub(crate) fn new(anchors: &[F], claim: F, degree: usize) -> Self {
        Self {
            anchors: anchors.to_vec(),
            claim,
            round: 0,
            degree,
            interpolator: RoundPolyInterpolator::new(5),
        }
    }

    /// The coefficients of the current round's cofactor above its constant.
    pub(crate) fn message(&self, round_poly: &[F; ROUND_POLY_LEN]) -> Vec<F> {
        let anchor = self.anchors[self.round];
        let values = [
            round_poly[0],
            self.claim - round_poly[0],
            round_poly[1],
            round_poly[2],
            round_poly[3],
            round_poly[4],
        ];
        let points = (0..=5)
            .map(F::interpolation_node)
            .zip(values)
            .filter_map(|(x, y)| {
                let line = equality_line(anchor, x);
                (!line.is_zero()).then(|| (x, y * line.inverse()))
            })
            .take(self.degree + 1)
            .collect::<Vec<_>>();
        assert_eq!(
            points.len(),
            self.degree + 1,
            "the equality line vanishes at one node at most"
        );
        let mut coefficients = coefficients_through(&points);
        coefficients.remove(0);
        coefficients
    }

    /// Bind the current round to `challenge`.
    pub(crate) fn bind(&mut self, round_poly: &[F; ROUND_POLY_LEN], challenge: F) {
        self.claim = self.interpolator.eval(round_poly, self.claim, challenge);
        self.round += 1;
    }
}

/// The coefficients, constant first, of the polynomial of least degree through `points`.
fn coefficients_through<F: Field>(points: &[(F, F)]) -> Vec<F> {
    // Newton's divided differences, then the nested form expanded from the top.
    let n = points.len();
    let mut differences = points.iter().map(|&(_, y)| y).collect::<Vec<_>>();
    for level in 1..n {
        for index in (level..n).rev() {
            let span = points[index].0 - points[index - level].0;
            differences[index] = (differences[index] - differences[index - 1]) * span.inverse();
        }
    }
    let mut coefficients = alloc::vec![F::ZERO; n];
    for index in (0..n).rev() {
        // coefficients <- coefficients * (X - x_index) + differences[index]
        let node = points[index].0;
        for k in (1..n).rev() {
            coefficients[k] = coefficients[k - 1] - node * coefficients[k];
        }
        coefficients[0] = differences[index] - node * coefficients[0];
    }
    coefficients
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoroshiro128Plus;

    use super::*;

    type EF = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn coefficients_through_recovers_a_polynomial() {
        let want = [3, 1, 4, 1, 5].map(EF::from_u32);
        let points = (0..5)
            .map(|i| {
                let x = EF::from_u32(7 * i + 2);
                (x, want.iter().rev().fold(EF::ZERO, |acc, &c| acc * x + c))
            })
            .collect::<Vec<_>>();
        assert_eq!(coefficients_through(&points), want);
    }

    fn factored_round_keeps_the_running_sum<F: Field>(g: [F; 5], anchor: F, scale: F, challenge: F) {
        // s(X) = scale * eq(anchor, X) * g(X), with g of degree four.
        let horner = |x: F| g.iter().rev().fold(F::ZERO, |acc, &c| acc * x + c);
        let s = |x: F| scale * equality_line(anchor, x) * horner(x);
        let claim = s(F::ZERO) + s(F::ONE);
        let round_poly = [0, 2, 3, 4, 5].map(|i| s(F::interpolation_node(i)));

        let mut factoring = Factoring::new(&[anchor], claim, 4);
        let message = factoring.message(&round_poly);
        assert_eq!(message, g[1..].iter().map(|&c| scale * c).collect::<Vec<_>>());
        assert_eq!(factored_step(&message, claim, anchor, challenge), s(challenge));
        factoring.bind(&round_poly, challenge);
        assert_eq!(factoring.claim, s(challenge));
    }

    #[test]
    fn a_factored_round_keeps_the_running_sum() {
        let ef = EF::from_u32;
        factored_round_keeps_the_running_sum([2, 7, 1, 8, 2].map(ef), ef(11), ef(5), ef(29));
        let mut rng = Xoroshiro128Plus::seed_from_u64(7);
        let mut b = || -> BinaryField128 { rng.random() };
        let g = [b(), b(), b(), b(), b()];
        factored_round_keeps_the_running_sum(g, b(), b(), b());
        // An anchor whose line vanishes at a transmitted node still leaves five points.
        let node = BinaryField128::interpolation_node(3);
        let anchor = (BinaryField128::ONE - node) * (BinaryField128::ONE - node.double()).inverse();
        assert!(equality_line(anchor, node).is_zero());
        factored_round_keeps_the_running_sum(g, anchor, b(), b());
    }
}
