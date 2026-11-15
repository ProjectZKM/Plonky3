//! The proof one Boolean opening carries.

use p3_binary_field::BitCoordinates;
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_sumcheck::ring_switch::bits::{
    BitRingSwitchClaimsProof, BitRingSwitchDeferredProof, TraceProof,
};
use p3_whir::{PairProof, PcsProof, WhirProof};
use serde::{Deserialize, Serialize};

/// One Boolean opening: the reductions, and the proximity opening that discharges them.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: BitCoordinates, EF: BitCoordinates, MT::Commitment: Serialize, MT::MultiProof: Serialize",
    deserialize = "F: BitCoordinates, EF: BitCoordinates, MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>"
))]
pub struct BooleanWhirProof<F: Field, EF: ExtensionField<F>, MT: Mmcs<F>> {
    /// One batched bit-alphabet ring switch, with every claim's elements in the order they came in.
    pub reduction: BitRingSwitchClaimsProof<F, EF>,
    /// The proximity opening that discharges the one surviving claim.
    pub opening: BooleanWhirOpening<F, EF, MT>,
}

/// How a Boolean opening's surviving claim is discharged.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: BitCoordinates, EF: BitCoordinates, MT::Commitment: Serialize, MT::MultiProof: Serialize",
    deserialize = "F: BitCoordinates, EF: BitCoordinates, MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>"
))]
pub enum BooleanWhirOpening<F: Field, EF: ExtensionField<F>, MT: Mmcs<F>> {
    /// A proximity opening of its own.
    Own(PcsProof<F, EF, MT>),
    /// The first side of a pair: the second side's proof carries the shared opening.
    Paired,
    /// The second side of a pair: one opening for both sides' surviving claims.
    Pair(PairProof<F, EF, MT>),
}

impl<F: Field, EF: ExtensionField<F>, MT: Mmcs<F>> BooleanWhirOpening<F, EF, MT> {
    /// The run that discharges this side's claim: its own, or the pair's.
    #[must_use]
    pub const fn whir(&self) -> Option<&WhirProof<F, EF, MT>> {
        match self {
            Self::Own(opening) => Some(&opening.whir),
            Self::Pair(pair) => Some(&pair.whir),
            Self::Paired => None,
        }
    }

    /// The run that discharges this side's claim, mutably.
    pub const fn whir_mut(&mut self) -> Option<&mut WhirProof<F, EF, MT>> {
        match self {
            Self::Own(opening) => Some(&mut opening.whir),
            Self::Pair(pair) => Some(&mut pair.whir),
            Self::Paired => None,
        }
    }
}

/// One Boolean opening whose ring switch defers its sum to the proximity argument's first fold.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: BitCoordinates, EF: BitCoordinates, MT::Commitment: Serialize, MT::MultiProof: Serialize",
    deserialize = "F: BitCoordinates, EF: BitCoordinates, MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>"
))]
pub struct BooleanWhirDeferredProof<F: Field, EF: ExtensionField<F>, MT: Mmcs<F>> {
    /// Every claim's elements, in the order they came in.
    pub reduction: BitRingSwitchDeferredProof<F, EF>,
    /// The proximity opening whose first fold proves the batch's sum.
    pub opening: WhirProof<F, EF, MT>,
}

/// A trace reduction of every opening, and the proximity opening whose first fold closes it.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: BitCoordinates, EF: BitCoordinates, MT::Commitment: Serialize, MT::MultiProof: Serialize",
    deserialize = "F: BitCoordinates, EF: BitCoordinates, MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>"
))]
pub struct BooleanWhirTraceProof<F: Field, EF: ExtensionField<F>, MT: Mmcs<F>> {
    /// One trace message per coordinate of the challenge field, for every opening at once.
    pub reduction: TraceProof<F>,
    /// The proximity opening whose first fold proves the surviving relation.
    pub opening: WhirProof<F, EF, MT>,
}
