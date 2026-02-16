//! Diffusion matrix for Bls12381
//!
//! Reference: https://github.com/0xEigenLabs/eigen-zkvm/blob/main/starky/src/poseidon_bls12381.rs

use std::sync::OnceLock;

use p3_field::FieldAlgebra;
use p3_poseidon2::{
    add_rc_and_sbox_generic, external_initial_permute_state, external_terminal_permute_state,
    internal_permute_state, matmul_internal, ExternalLayer, ExternalLayerConstants,
    ExternalLayerConstructor, HLMDSMat4, InternalLayer, InternalLayerConstructor, Poseidon2,
};
use serde::{Deserialize, Serialize};

use crate::Bls12381Fr;

/// Degree of the chosen permutation polynomial for BLS12381, used as the Poseidon2 S-Box.
///
/// As p - 1 is divisible by 2 and 3 the smallest choice for a degree D satisfying gcd(p - 1, D) = 1 is 5.
const BLS12381_S_BOX_DEGREE: u64 = 5;

/// An implementation of the Poseidon2 hash function for the Bls12381Fr field.
///
/// It acts on arrays of the form `[Bls12381Fr; WIDTH]`.
pub type Poseidon2Bls12381<const WIDTH: usize> = Poseidon2<
    Bls12381Fr,
    Poseidon2ExternalLayerBls12381<WIDTH>,
    Poseidon2InternalLayerBls12381,
    WIDTH,
    BLS12381_S_BOX_DEGREE,
>;

/// Currently we only support a single width for Poseidon2 BLS12381.
const BLS12381_WIDTH: usize = 3;

#[inline]
fn get_diffusion_matrix_3() -> &'static [Bls12381Fr; 3] {
    static MAT_DIAG3_M_1: OnceLock<[Bls12381Fr; 3]> = OnceLock::new();
    MAT_DIAG3_M_1.get_or_init(|| [Bls12381Fr::ONE, Bls12381Fr::ONE, Bls12381Fr::TWO])
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Poseidon2InternalLayerBls12381 {
    internal_constants: Vec<Bls12381Fr>,
}

impl InternalLayerConstructor<Bls12381Fr> for Poseidon2InternalLayerBls12381 {
    fn new_from_constants(internal_constants: Vec<Bls12381Fr>) -> Self {
        Self { internal_constants }
    }
}

impl InternalLayer<Bls12381Fr, BLS12381_WIDTH, BLS12381_S_BOX_DEGREE>
    for Poseidon2InternalLayerBls12381
{
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [Bls12381Fr; BLS12381_WIDTH]) {
        internal_permute_state::<Bls12381Fr, BLS12381_WIDTH, BLS12381_S_BOX_DEGREE>(
            state,
            |x| matmul_internal(x, *get_diffusion_matrix_3()),
            &self.internal_constants,
        )
    }
}

pub type Poseidon2ExternalLayerBls12381<const WIDTH: usize> =
    ExternalLayerConstants<Bls12381Fr, WIDTH>;

impl<const WIDTH: usize> ExternalLayerConstructor<Bls12381Fr, WIDTH>
    for Poseidon2ExternalLayerBls12381<WIDTH>
{
    fn new_from_constants(external_constants: ExternalLayerConstants<Bls12381Fr, WIDTH>) -> Self {
        external_constants
    }
}

impl<const WIDTH: usize> ExternalLayer<Bls12381Fr, WIDTH, BLS12381_S_BOX_DEGREE>
    for Poseidon2ExternalLayerBls12381<WIDTH>
{
    /// Perform the initial external layers of the Poseidon2 permutation on the given state.
    fn permute_state_initial(&self, state: &mut [Bls12381Fr; WIDTH]) {
        external_initial_permute_state(
            state,
            self.get_initial_constants(),
            add_rc_and_sbox_generic::<_, BLS12381_S_BOX_DEGREE>,
            &HLMDSMat4,
        );
    }

    /// Perform the terminal external layers of the Poseidon2 permutation on the given state.
    fn permute_state_terminal(&self, state: &mut [Bls12381Fr; WIDTH]) {
        external_terminal_permute_state(
            state,
            self.get_terminal_constants(),
            add_rc_and_sbox_generic::<_, BLS12381_S_BOX_DEGREE>,
            &HLMDSMat4,
        );
    }
}

#[cfg(test)]
mod tests {
    use ff::PrimeField;
    use p3_poseidon2::ExternalLayerConstants;
    use p3_symmetric::Permutation;
    use rand::Rng;
    use zkhash::ark_ff::{BigInteger, PrimeField as ark_PrimeField};
    use zkhash::fields::bls12::FpBLS12 as ark_FpBLS12;
    use zkhash::poseidon2::poseidon2::Poseidon2 as Poseidon2Ref;
    // TODO: should test 2, 3, 4, 8
    use zkhash::poseidon2::poseidon2_instance_bls12::{POSEIDON2_BLS_3_PARAMS, RC3};

    use super::*;
    use crate::FFBls12381Fr;

    fn bls12_from_ark_ff(input: ark_FpBLS12) -> Bls12381Fr {
        let bytes = input.into_bigint().to_bytes_le();

        let mut res = <FFBls12381Fr as PrimeField>::Repr::default();

        for (i, digit) in res.as_mut().iter_mut().enumerate() {
            *digit = bytes[i];
        }

        let value = FFBls12381Fr::from_repr(res);

        if value.is_some().into() {
            Bls12381Fr {
                value: value.unwrap(),
            }
        } else {
            panic!("Invalid field element")
        }
    }

    #[test]
    fn test_poseidon2_bls12381() {
        const WIDTH: usize = 3;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 56;

        type F = Bls12381Fr;

        let mut rng = rand::thread_rng();

        // Poiseidon2 reference implementation from zkhash repo.
        let poseidon2_ref = Poseidon2Ref::new(&POSEIDON2_BLS_3_PARAMS);

        // Copy over round constants from zkhash.
        let mut round_constants: Vec<[F; WIDTH]> = RC3
            .iter()
            .map(|vec| {
                vec.iter()
                    .cloned()
                    .map(bls12_from_ark_ff)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect();

        let internal_start = ROUNDS_F / 2;
        let internal_end = (ROUNDS_F / 2) + ROUNDS_P;
        let internal_round_constants = round_constants
            .drain(internal_start..internal_end)
            .map(|vec| vec[0])
            .collect::<Vec<_>>();
        let external_round_constants = ExternalLayerConstants::new(
            round_constants[..(ROUNDS_F / 2)].to_vec(),
            round_constants[(ROUNDS_F / 2)..].to_vec(),
        );
        // Our Poseidon2 implementation.
        let poseidon2 = Poseidon2Bls12381::new(external_round_constants, internal_round_constants);

        // Generate random input and convert to both Goldilocks field formats.
        let input_ark_ff = rng.gen::<[ark_FpBLS12; WIDTH]>();
        let input: [Bls12381Fr; 3] = input_ark_ff
            .iter()
            .cloned()
            .map(bls12_from_ark_ff)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Run reference implementation.
        let output_ref = poseidon2_ref.permutation(&input_ark_ff);

        let expected: [F; WIDTH] = output_ref
            .iter()
            .cloned()
            .map(bls12_from_ark_ff)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Run our implementation.
        let mut output = input;
        poseidon2.permute_mut(&mut output);

        assert_eq!(output, expected);
    }
}
