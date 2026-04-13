//! The scalar field of the BN254 curve, defined as `F_P` where `P = 21888242871839275222246405745257275088548364400416034343698204186575808495617`.
#![no_std]

extern crate alloc;

mod bn254;
mod helpers;
mod poseidon2;

pub use bn254::*;
pub use poseidon2::Poseidon2Bn254;

/// Backward-compatible alias: main branch uses `Bn254Fr` for the scalar field type.
pub type Bn254Fr = Bn254;
