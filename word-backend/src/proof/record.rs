//! The transcript record one complete word-level proof carries.

use p3_sumcheck::generic_degree::GenericDegreeProof;
use serde::{Deserialize, Serialize};

use super::relation::OPERAND_EVALUATIONS;
use crate::ShiftReductionProof;
use crate::PortProof;
use crate::integer_mul::IntegerMulProof;

/// Transcript record of one complete word-level proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WordProof<F, EF, Opening> {
    /// Reduction of every unsigned product, present exactly when the statement declares one.
    pub(super) integer_mul: Option<IntegerMulProof<F, EF>>,
    /// Rounds of the batched relation vanishing check.
    pub(super) zerocheck: GenericDegreeProof<F, EF>,
    /// Operand evaluations left at the point the vanishing check ends on.
    pub(super) operands: [EF; OPERAND_EVALUATIONS],
    /// Record of the reduction from shifted operands to one trace claim.
    pub(super) shift: ShiftReductionProof<F, EF>,
    /// Reduction of the copy constraints between committed ports, present exactly when the key
    /// routes ports.
    pub(super) ports: Option<PortProof<F, EF>>,
    /// Proof that the committed trace takes the reduced value at the reduced point.
    pub(super) opening: Opening,
    /// Proof that the committed routing words take the port reduction's value at its point,
    /// present exactly when the key routes ports.
    pub(super) port_opening: Option<Opening>,
}

impl<F, EF, Opening> WordProof<F, EF, Opening> {
    /// The proof that the committed trace takes the reduced value.
    pub const fn opening(&self) -> &Opening {
        &self.opening
    }

    /// The proof that the committed routing words take the port reduction's value, when the
    /// key routes ports.
    pub const fn port_opening(&self) -> Option<&Opening> {
        self.port_opening.as_ref()
    }

    /// The reduction of the copy constraints between committed ports, when the key routes
    /// ports.
    pub const fn ports(&self) -> Option<&PortProof<F, EF>> {
        self.ports.as_ref()
    }
}
