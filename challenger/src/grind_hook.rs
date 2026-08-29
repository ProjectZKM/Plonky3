//! Pluggable accelerator for the duplex-sponge proof-of-work grind.
//!
//! The grind is a brute-force scan — for each candidate witness `w`, one
//! permutation of `[input_buffer || w || sponge_state..]` and a check that
//! the low `bits` of the sampled element are zero — which a GPU does orders
//! of magnitude faster than the host SIMD scan.  A prover that has a device
//! available installs a hook here once; `DuplexChallenger::grind` consults
//! it first and falls back to the windowed host scan whenever the hook is
//! absent, declines (`None`), or the sponge shape differs from what the
//! hook supports.
//!
//! The contract is EXACT equivalence with the host scan:
//! * candidates are enumerated in canonical order `w = 0, 1, 2, ...` and
//!   the SMALLEST passing `w` must be returned;
//! * the state layout is `state[i] = input_buffer[i]` for
//!   `i < input_buffer.len()`, the candidate at `witness_idx =
//!   input_buffer.len()`, and `sponge_state[i]` above that;
//! * the predicate is `canonical(permuted[rate - 1]) & ((1 << bits) - 1)
//!   == 0`.
//! `grind` re-verifies whatever the hook returns with `check_witness`, so a
//! wrong hook fails loudly rather than corrupting a transcript.
//!
//! Values cross the hook boundary as CANONICAL `u64`s so the hook stays
//! field-agnostic; the installer is expected to check the state width and
//! `rate` and decline shapes it does not implement.

use alloc::boxed::Box;
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

/// The hook: `(sponge_state, input_buffer, output_buffer, bits) ->
/// Option<witness>`.
///
/// The three slices are the challenger's own duplex buffers in canonical
/// form, exactly as `DuplexChallenger` holds them at the call — the hook
/// replays `observe(candidate)` + `sample_bits(bits)` itself, so it needs
/// them separately rather than pre-merged.  It must NOT advance anything:
/// `grind` applies the transcript mutation through `check_witness` on the
/// value the hook returns.
pub type GrindHookFn =
    dyn Fn(&[u64], &[u64], &[u64], usize) -> Option<u64> + Send + Sync + 'static;

static HOOK: AtomicPtr<Box<GrindHookFn>> = AtomicPtr::new(ptr::null_mut());

/// Install the accelerator.  First call wins; later calls are dropped.
pub fn install(hook: Box<GrindHookFn>) {
    let boxed = Box::into_raw(Box::new(hook));
    if HOOK
        .compare_exchange(ptr::null_mut(), boxed, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        // Lost the race: reclaim the wrapper we made.
        drop(unsafe { Box::from_raw(boxed) });
    }
}

/// The installed accelerator, if any.
pub fn get() -> Option<&'static GrindHookFn> {
    let p = HOOK.load(Ordering::Acquire);
    if p.is_null() {
        None
    } else {
        // SAFETY: a non-null pointer was leaked by `install` and is never
        // freed, so the reference lives for the rest of the process.
        Some(unsafe { &**p })
    }
}
