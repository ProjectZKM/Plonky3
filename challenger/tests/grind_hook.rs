//! The grind hook contract, checked against the host scan.
//!
//! This lives in its own test binary: the hook is process-global and first-install-wins.

use std::sync::atomic::{AtomicUsize, Ordering};

use p3_baby_bear::{BabyBear, Poseidon2BabyBear, default_babybear_poseidon2_16};
use p3_challenger::{CanObserve, DuplexChallenger, GrindingChallenger, grind_hook};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_symmetric::Permutation;

type F = BabyBear;
type Perm = Poseidon2BabyBear<16>;
type Challenger = DuplexChallenger<F, Perm, 16, 8>;

const WIDTH: usize = 16;
const RATE: usize = 8;

static HOOK_CALLS: AtomicUsize = AtomicUsize::new(0);

/// A scalar implementation of the documented hook contract.
fn contract_hook(sponge: &[u64], input: &[u64], _output: &[u64], bits: usize) -> Option<u64> {
    if sponge.len() != WIDTH || input.len() >= RATE {
        return None;
    }
    HOOK_CALLS.fetch_add(1, Ordering::Relaxed);
    let perm = default_babybear_poseidon2_16();
    let witness_idx = input.len();
    let mut base = [F::ZERO; WIDTH];
    for (i, &v) in input.iter().enumerate() {
        base[i] = F::from_u64(v);
    }
    base[RATE] = F::from_u64(sponge[RATE]) + F::from_usize(witness_idx + 1);
    for i in RATE + 1..WIDTH {
        base[i] = F::from_u64(sponge[i]);
    }
    let mask = (1u64 << bits) - 1;
    (0..F::ORDER_U64).find(|&w| {
        let mut state = base;
        state[witness_idx] = F::from_u64(w);
        perm.permute_mut(&mut state);
        state[RATE - 1].as_canonical_u64() & mask == 0
    })
}

fn smallest_passing_witness(challenger: &Challenger, bits: usize) -> F {
    (0..F::ORDER_U64)
        .map(F::from_u64)
        .find(|&w| challenger.clone().check_witness(bits, w))
        .unwrap()
}

#[test]
fn hook_matches_host_scan() {
    grind_hook::install(Box::new(contract_hook));

    for num_observed in [0, 1, 3, 7, 8, 9] {
        let mut challenger = Challenger::new(default_babybear_poseidon2_16());
        for i in 0..num_observed {
            challenger.observe(F::from_u32(i as u32 + 1));
        }
        let bits = 8;
        let expected = smallest_passing_witness(&challenger, bits);

        let calls_before = HOOK_CALLS.load(Ordering::Relaxed);
        let witness = challenger.grind(bits);
        assert_eq!(witness, expected, "num_observed = {num_observed}");
        assert_eq!(HOOK_CALLS.load(Ordering::Relaxed), calls_before + 1);
    }
}
