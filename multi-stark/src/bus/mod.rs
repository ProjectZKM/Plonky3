//! Binary-native bus planning and commitment binding.

pub(crate) mod composition;

pub use composition::{BusFamilyDevice, BusTermsSpec, DeviceBusTerms};
mod context;
mod error;

pub(crate) use context::BusContext;
pub use context::BusLeafBlock;
pub use error::BusBindingError;
