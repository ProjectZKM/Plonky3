//! WHIR polynomial commitment scheme: commit, prove, and verify.

mod adapter;
pub(crate) mod committer;
pub mod device;
pub mod pair;
pub mod proof;
pub mod prover;
mod security;
pub mod utils;
pub mod verifier;
pub mod zk;

pub use adapter::WhirProverData;
pub use device::{
    BaseProverData, Device, ExtProverData, PairDevice, PairDevices, RoundDevice, SingleDevices,
};
pub use pair::{PairProof, pair_batching_error};

#[cfg(test)]
mod tests;
