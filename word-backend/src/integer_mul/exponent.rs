//! The exponent lift: the bits of an integer select powers of one multiplicative generator.
//!
//! ```text
//!     g        a generator of the challenge field's multiplicative group
//!     G_t      g^(2^t), by repeated squaring
//!
//!     prod_t (1 + x_t * (G_t - 1))  =  prod_{t : x_t = 1} G_t  =  g^(sum_t x_t * 2^t)  =  g^x
//! ```
//!
//! Every leaf factor is one or the power its bit selects, so a product of leaves is a power of `g`.
//!
//! The power tree raises the lift of one factor by the bits of the other the same way.

use alloc::vec::Vec;
use core::iter;

use p3_field::Field;
use p3_multilinear_util::point::Point;

/// Returns `g^(2^t)` for every `t < count`.
pub(super) fn generator_squarings<F: Field>(count: usize) -> Vec<F> {
    // Squaring doubles the exponent, so the walk visits consecutive powers of two.
    iter::successors(Some(F::GENERATOR), |power| Some(power.square()))
        .take(count)
        .collect()
}

/// Evaluates the multilinear extension of a table at a point.
pub(super) fn evaluate<F: Field>(table: &[F], point: &[F]) -> F {
    // The first coordinate addresses the most significant index bit.
    debug_assert_eq!(table.len(), 1 << point.len());
    F::batched_linear_combination(&Point::new(point).equality_weights_msb(), table)
}

#[cfg(test)]
mod tests {
    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;

    type F = BinaryField128;

    // The lift of an integer below 2^128, split into two exponentiations by 64-bit halves.
    fn lift(value: u128) -> F {
        let high = F::GENERATOR
            .exp_power_of_2(64)
            .exp_u64((value >> 64) as u64);
        F::GENERATOR.exp_u64(value as u64) * high
    }

    // The product of the selected leaves of one word, bit t selecting powers[t].
    fn selected(word: u128, powers: &[F]) -> F {
        (0..powers.len())
            .filter(|&bit| (word >> bit) & 1 == 1)
            .map(|bit| powers[bit])
            .product()
    }

    // The power-tree root of one row: the base lift raised by the right factor's bits.
    fn power_root(a: u64, b: u64) -> F {
        let base = selected(a.into(), &generator_squarings::<F>(64));
        let powers = core::iter::successors(Some(base), |power| Some(power.square()))
            .take(64)
            .collect::<Vec<_>>();
        selected(b.into(), &powers)
    }

    #[test]
    fn the_squarings_double_the_exponent() {
        let squarings = generator_squarings::<F>(8);
        assert_eq!(squarings[0], F::GENERATOR);
        assert_eq!(squarings[7], F::GENERATOR.exp_power_of_2(7));
    }

    #[test]
    fn the_wraparound_is_the_one_collision_of_a_128_bit_lift() {
        // The multiplicative group has order 2^128 - 1, so that exponent lifts to one.
        assert_eq!(lift(u128::MAX), F::ONE);

        // A zero factor also lifts to one.
        //
        // Claiming both limbs all-ones for a zero product therefore passes the lift alone.
        assert_eq!(power_root(0, 17), F::ONE);
        assert_eq!(selected(u128::MAX, &generator_squarings::<F>(128)), F::ONE);
    }

    proptest! {
        #[test]
        fn both_trees_lift_their_integer(a: u64, b: u64) {
            // Invariant: each root is g raised to the integer its bits encode.
            let product = u128::from(a) * u128::from(b);
            prop_assert_eq!(power_root(a, b), lift(product));
            prop_assert_eq!(selected(product, &generator_squarings::<F>(128)), lift(product));
        }

        #[test]
        fn a_wrong_limb_moves_the_result_root(a: u64, b: u64, flip in 0usize..128) {
            // Mutation: flip one bit of the honest limbs.
            //
            // The lift is injective below 2^128 - 1, so the roots must now differ.
            let product = u128::from(a) * u128::from(b);
            let wrong = product ^ (1u128 << flip);
            prop_assume!(wrong != u128::MAX);
            prop_assert_ne!(selected(wrong, &generator_squarings::<F>(128)), power_root(a, b));
        }
    }
}
