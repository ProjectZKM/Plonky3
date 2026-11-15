//! Word-level relation proving, from a packed witness to one authenticated trace opening.
//!
//! A key proves either a flat constraint system or a composition of repeated gadgets.
//!
//! The composed form compiles each gadget once, whatever its instance count.
//!
//! Its stored wiring is therefore the size of the gadgets, not the size of the statement.
//!
//! Both forms reach the same proof.

#![no_std]

extern crate alloc;

mod columns;
mod device;
mod integer_mul;
mod keys;
mod ports;
mod proof;
mod shift;
mod statement;
mod witness;

pub use columns::OperationColumns;
pub use device::{
    DevicePorts, DeviceRelation, DeviceSelector, DeviceShift, DeviceTree, OPERATION_COLUMNS, PRODUCT_COLUMNS, LINEAR_FAMILY, BITWISE_FAMILY, PRODUCT_FAMILY,
    ShiftLayout, WordDevice, Words, shift_kind_code,
};
pub use integer_mul::IntegerMulError;
pub use keys::{
    CompiledKey, CompiledKeyLayout, CompiledSegment, ConstraintReference, KeyCompileError,
    LayoutComponent, LayoutFootprint,
};
pub use ports::{PortError, PortProof, PortProofError, PortRef, PortRouting};
pub use proof::{WordProof, WordProofError, WordProofKey};
pub use shift::{
    Deferred, ShiftClaim, ShiftHints, ShiftOpeningClaim, ShiftReductionError, ShiftReductionKey,
    ShiftReductionProof, WiringQuery,
};
pub use statement::{Statement, StatementShape};
pub use witness::{Packed, PackedWitness, PackedWord, WitnessError};
