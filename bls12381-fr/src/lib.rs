//! The scalar field of the BLS12381 curve, defined as `F_r` where `r = 52435875175126190479447740508185965837690552500527637822603658699938581184513`.

mod poseidon2;

use core::fmt;
use core::fmt::{Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use ff::{Field as FFField, PrimeField as FFPrimeField};
pub use halo2curves::bls12381::Fr as FFBls12381Fr;
use halo2curves::serde::SerdeObject;
use num_bigint::BigUint;
use p3_field::{Field, FieldAlgebra, Packable, PrimeField, TwoAdicField};
pub use poseidon2::Poseidon2Bls12381;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Deserializer, Serialize};

/// The BLS12381 curve scalar field prime, defined as `F_r` where `r = 52435875175126190479447740508185965837690552500527637822603658699938581184513`.
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct Bls12381Fr {
    pub value: FFBls12381Fr,
}

impl Bls12381Fr {
    pub(crate) const fn new(value: FFBls12381Fr) -> Self {
        Self { value }
    }
}

impl Serialize for Bls12381Fr {
    /// Serializes to raw bytes, which are typically of the Montgomery representation of the field element.
    // See https://github.com/privacy-scaling-explorations/halo2curves/blob/d34e9e46f7daacd194739455de3b356ca6c03206/derive/src/field/mod.rs#L493
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let bytes = self.value.to_raw_bytes();
        serializer.serialize_bytes(&bytes)
    }
}

impl<'de> Deserialize<'de> for Bls12381Fr {
    /// Deserializes from raw bytes, which are typically of the Montgomery representation of the field element.
    /// Performs a check that the deserialized field element corresponds to a value less than the field modulus, and
    /// returns error otherwise.
    // See https://github.com/privacy-scaling-explorations/halo2curves/blob/d34e9e46f7daacd194739455de3b356ca6c03206/derive/src/field/mod.rs#L485
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let bytes: Vec<u8> = Deserialize::deserialize(d)?;

        let value = FFBls12381Fr::from_raw_bytes(&bytes);

        value
            .map(Self::new)
            .ok_or(serde::de::Error::custom("Invalid field element"))
    }
}

impl Packable for Bls12381Fr {}

impl Hash for Bls12381Fr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for byte in self.value.to_repr().as_ref().iter() {
            state.write_u8(*byte);
        }
    }
}

impl Ord for Bls12381Fr {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl PartialOrd for Bls12381Fr {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for Bls12381Fr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <FFBls12381Fr as Debug>::fmt(&self.value, f)
    }
}

impl Debug for Bls12381Fr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.value, f)
    }
}

impl FieldAlgebra for Bls12381Fr {
    type F = Self;

    const ZERO: Self = Self::new(FFBls12381Fr::ZERO);
    const ONE: Self = Self::new(FFBls12381Fr::ONE);
    const TWO: Self = Self::new(FFBls12381Fr::from_raw([2u64, 0, 0, 0]));

    // r - 1 = 0x73eda753299d7d48 3339d80809a1d805 53bda402fffe5bfe ffffffff00000000
    const NEG_ONE: Self = Self::new(FFBls12381Fr::from_raw([
        0xffffffff00000000,
        0x53bda402fffe5bfe,
        0x3339d80809a1d805,
        0x73eda753299d7d48,
    ]));

    #[inline]
    fn from_f(f: Self::F) -> Self {
        f
    }

    fn from_bool(b: bool) -> Self {
        Self::new(FFBls12381Fr::from(b as u64))
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self::new(FFBls12381Fr::from(n as u64))
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self::new(FFBls12381Fr::from(n as u64))
    }

    fn from_canonical_u32(n: u32) -> Self {
        Self::new(FFBls12381Fr::from(n as u64))
    }

    fn from_canonical_u64(n: u64) -> Self {
        Self::new(FFBls12381Fr::from(n))
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self::new(FFBls12381Fr::from(n as u64))
    }

    fn from_wrapped_u32(n: u32) -> Self {
        Self::new(FFBls12381Fr::from(n as u64))
    }

    fn from_wrapped_u64(n: u64) -> Self {
        Self::new(FFBls12381Fr::from(n))
    }
}

impl Field for Bls12381Fr {
    type Packing = Self;

    // generator is 7
    const GENERATOR: Self = Self::new(FFBls12381Fr::from_raw([7u64, 0, 0, 0]));

    fn is_zero(&self) -> bool {
        self.value.is_zero().into()
    }

    fn try_inverse(&self) -> Option<Self> {
        let inverse = self.value.invert();

        if inverse.is_some().into() {
            Some(Self::new(inverse.unwrap()))
        } else {
            None
        }
    }

    /// r = 0x73eda753_299d7d48_3339d808_09a1d805_53bda402_fffe5bfe_ffffffff_00000001
    fn order() -> BigUint {
        BigUint::new(vec![
            0x00000001, 0xffffffff, 0xfffe5bfe, 0x53bda402, 0x09a1d805, 0x3339d808, 0x299d7d48,
            0x73eda753,
        ])
    }

    // https://github.com/docknetwork/crypto/blob/main/vb_accumulator/src/universal_init.sage#L81
    fn multiplicative_group_factors() -> Vec<(BigUint, usize)> {
        vec![
            (BigUint::from(2u8), 32),
            (BigUint::from(3u8), 1),
            (BigUint::from(11u8), 1),
            (BigUint::from(19u8), 1),
            (BigUint::from(10177u16), 1),
            (BigUint::from(125527u32), 1),
            (BigUint::from(859267u32), 1),
            (BigUint::from(906349u32), 2),
            (BigUint::from(2508409u32), 1),
            (BigUint::from(2529403u32), 1),
            (BigUint::from(52437899u32), 1),
            (BigUint::from(254760293u32), 2),
        ]
    }
}

impl PrimeField for Bls12381Fr {
    fn as_canonical_biguint(&self) -> BigUint {
        let repr = self.value.to_repr();
        let le_bytes = repr.as_ref();
        BigUint::from_bytes_le(le_bytes)
    }
}

impl Add for Bls12381Fr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.value + rhs.value)
    }
}

impl AddAssign for Bls12381Fr {
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl Sum for Bls12381Fr {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::ZERO)
    }
}

impl Sub for Bls12381Fr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.value.sub(rhs.value))
    }
}

impl SubAssign for Bls12381Fr {
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl Neg for Bls12381Fr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * Self::NEG_ONE
    }
}

impl Mul for Bls12381Fr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new(self.value * rhs.value)
    }
}

impl MulAssign for Bls12381Fr {
    fn mul_assign(&mut self, rhs: Self) {
        self.value *= rhs.value;
    }
}

impl Product for Bls12381Fr {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::ONE)
    }
}

impl Div for Bls12381Fr {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

impl Distribution<Bls12381Fr> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Bls12381Fr {
        Bls12381Fr::new(FFBls12381Fr::random(rng))
    }
}

impl TwoAdicField for Bls12381Fr {
    const TWO_ADICITY: usize = FFBls12381Fr::S as usize;

    fn two_adic_generator(bits: usize) -> Self {
        let mut omega = FFBls12381Fr::ROOT_OF_UNITY;
        for _ in bits..Self::TWO_ADICITY {
            omega = omega.square();
        }
        Self::new(omega)
    }
}

#[cfg(test)]
mod tests {
    use num_traits::One;
    use p3_field_testing::test_field;

    use super::*;

    type F = Bls12381Fr;

    #[test]
    fn test_bls12381fr() {
        let f = F::new(FFBls12381Fr::from_u128(100));
        assert_eq!(f.as_canonical_biguint(), BigUint::new(vec![100]));

        let f = F::from_canonical_u64(0);
        assert!(f.is_zero());

        let f = F::new(FFBls12381Fr::from_str_vartime(&F::order().to_str_radix(10)).unwrap());
        assert!(f.is_zero());

        assert_eq!(F::GENERATOR.as_canonical_biguint(), BigUint::new(vec![7]));

        let f_1 = F::new(FFBls12381Fr::from_u128(1));
        let f_1_copy = F::new(FFBls12381Fr::from_u128(1));

        let expected_result = F::ZERO;
        assert_eq!(f_1 - f_1_copy, expected_result);

        let expected_result = F::new(FFBls12381Fr::from_u128(2));
        assert_eq!(f_1 + f_1_copy, expected_result);

        let f_2 = F::new(FFBls12381Fr::from_u128(2));
        let expected_result = F::new(FFBls12381Fr::from_u128(3));
        assert_eq!(f_1 + f_1_copy * f_2, expected_result);

        let expected_result = F::new(FFBls12381Fr::from_u128(5));
        assert_eq!(f_1 + f_2 * f_2, expected_result);

        let f_r_minus_1 = F::new(
            FFBls12381Fr::from_str_vartime(&(F::order() - BigUint::one()).to_str_radix(10))
                .unwrap(),
        );
        let expected_result = F::ZERO;
        assert_eq!(f_1 + f_r_minus_1, expected_result);

        let f_r_minus_2 = F::new(
            FFBls12381Fr::from_str_vartime(&(F::order() - BigUint::new(vec![2])).to_str_radix(10))
                .unwrap(),
        );
        let expected_result = F::new(
            FFBls12381Fr::from_str_vartime(&(F::order() - BigUint::new(vec![3])).to_str_radix(10))
                .unwrap(),
        );
        assert_eq!(f_r_minus_1 + f_r_minus_2, expected_result);

        let expected_result = F::new(FFBls12381Fr::from_u128(1));
        assert_eq!(f_r_minus_1 - f_r_minus_2, expected_result);

        let expected_result = f_r_minus_1;
        assert_eq!(f_r_minus_2 - f_r_minus_1, expected_result);

        let expected_result = f_r_minus_2;
        assert_eq!(f_r_minus_1 - f_1, expected_result);

        let expected_result = F::new(FFBls12381Fr::from_u128(3));
        assert_eq!(f_2 * f_2 - f_1, expected_result);

        // Generator check
        let expected_multiplicative_group_generator = F::new(FFBls12381Fr::from_u128(7));
        assert_eq!(F::GENERATOR, expected_multiplicative_group_generator);

        let f_serialized = serde_json::to_string(&f).unwrap();
        let f_deserialized: F = serde_json::from_str(&f_serialized).unwrap();
        assert_eq!(f, f_deserialized);

        let f_1_serialized = serde_json::to_string(&f_1).unwrap();
        let f_1_deserialized: F = serde_json::from_str(&f_1_serialized).unwrap();
        let f_1_serialized_again = serde_json::to_string(&f_1_deserialized).unwrap();
        let f_1_deserialized_again: F = serde_json::from_str(&f_1_serialized_again).unwrap();
        assert_eq!(f_1, f_1_deserialized);
        assert_eq!(f_1, f_1_deserialized_again);

        let f_2_serialized = serde_json::to_string(&f_2).unwrap();
        let f_2_deserialized: F = serde_json::from_str(&f_2_serialized).unwrap();
        assert_eq!(f_2, f_2_deserialized);

        let f_r_minus_1_serialized = serde_json::to_string(&f_r_minus_1).unwrap();
        let f_r_minus_1_deserialized: F = serde_json::from_str(&f_r_minus_1_serialized).unwrap();
        assert_eq!(f_r_minus_1, f_r_minus_1_deserialized);

        let f_r_minus_2_serialized = serde_json::to_string(&f_r_minus_2).unwrap();
        let f_r_minus_2_deserialized: F = serde_json::from_str(&f_r_minus_2_serialized).unwrap();
        assert_eq!(f_r_minus_2, f_r_minus_2_deserialized);
    }

    test_field!(crate::Bls12381Fr);
}
