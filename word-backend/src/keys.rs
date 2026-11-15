//! Compiled word-to-relation metadata for backend reductions.

use alloc::vec::Vec;
use core::ops::Range;

use p3_word::{
    ComponentCall, Composition, ConstraintKind, ConstraintSystem, OperandRole, Placement,
    Segment, Shift, Word,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod builder;
mod code;

use code::ShiftSequenceCode;

/// Number of relation families a statement can declare.
const FAMILIES: usize = 3;

/// A compact index into one operation family.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ConstraintReference {
    /// The operand's semantic role within its relation.
    operand: OperandRole,
    /// The position within its homogeneous relation family.
    constraint: u32,
}

impl ConstraintReference {
    /// Returns the operand's semantic role within the relation.
    #[inline]
    pub const fn operand(self) -> OperandRole {
        self.operand
    }

    /// Returns the constraint position within the operation family.
    #[inline]
    pub const fn constraint(self) -> u32 {
        self.constraint
    }
}

/// One word's resolved references under a fixed operation and shift sequence.
///
/// The stored references are the ones one gadget slot carries.
///
/// An instanced segment stores them once and adds the instance's offset on the way out.
///
/// That is what keeps the stored metadata independent of the instance count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledKey<'a, W: Word> {
    /// The relation family consuming the word.
    operation: ConstraintKind,
    /// The dense index of the shift sequence.
    shift_index: u32,
    /// The inner and outer movements in evaluation order.
    shifts: [Shift<W>; 2],
    /// The relation occurrences consuming the shifted word, in slot-local form.
    references: &'a [ConstraintReference],
    /// The instance's first relation position within this key's family.
    constraint_offset: u32,
    /// The power of two a slot-local relation position is scaled by: the instance count of an
    /// interleaved block, one otherwise.
    constraint_shift: u8,
}

impl<W: Word> CompiledKey<'_, W> {
    /// Returns the relation family using the word.
    #[inline]
    pub const fn operation(&self) -> ConstraintKind {
        self.operation
    }

    /// Returns the dense shift-sequence index.
    #[inline]
    pub const fn shift_index(&self) -> u32 {
        self.shift_index
    }

    /// Returns the inner and outer movements in evaluation order.
    #[inline]
    pub const fn shifts(&self) -> [Shift<W>; 2] {
        self.shifts
    }

    /// Returns the number of relation occurrences grouped under this key.
    #[inline]
    pub const fn num_references(&self) -> usize {
        self.references.len()
    }

    /// Returns every relation occurrence in reduction order.
    ///
    /// Positions are already resolved to the composed statement.
    #[inline]
    pub fn references(
        &self,
    ) -> impl ExactSizeIterator<Item = ConstraintReference> + Clone + use<'_, W> {
        let offset = self.constraint_offset;
        let shift = self.constraint_shift;
        self.references.iter().map(move |reference| {
            // The stored position is slot-local, so the instance is added here.
            ConstraintReference {
                operand: reference.operand,
                constraint: (reference.constraint << shift) + offset,
            }
        })
    }
}

/// One stored key before its references are resolved to borrowed slices.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct StoredKey {
    /// The relation family consuming the word.
    operation: ConstraintKind,
    /// The dense index of the shift sequence.
    shift: u32,
    /// The contiguous span of relation references.
    references: Range<u32>,
}

/// One call's slot storage, repeated across its instances.
///
/// A flat statement is the degenerate case.
///
/// It is one block of one instance whose slots are the segment's own words.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct InstanceBlock {
    /// The block's first word in the composed segment.
    word_base: u32,
    /// Words one instance occupies, which is also the instance stride.
    slots: u32,
    /// Live instances of the call.
    instances: u32,
    /// The block's first entry in the shared slot-key table.
    slot_base: u32,
    /// Relations one instance adds to each family, in family-code order.
    strides: [u32; FAMILIES],
    /// The first relation position of instance zero, in family-code order.
    constraint_base: [u32; FAMILIES],
    /// Whether the instance index is the low bits of the block's words, the instance count
    /// then a power of two.
    #[serde(default)]
    interleaved: bool,
    /// Whether the instance index is the low bits of the block's relations, the instance count
    /// then a power of two.
    #[serde(default)]
    scaled: bool,
}

impl InstanceBlock {
    /// Returns the number of words the block spans.
    #[inline]
    const fn width(&self) -> u32 {
        // Both factors were bounded when the composition was laid out.
        self.slots * self.instances
    }

    /// Returns the instance and slot a composed word belongs to.
    #[inline]
    const fn split(&self, word: u32) -> (u32, u32) {
        let local = word - self.word_base;
        if self.interleaved {
            return (local & (self.instances - 1), local >> self.log_instances());
        }
        (local / self.slots, local % self.slots)
    }

    /// The base-two logarithm of an interleaved block's instance count; zero otherwise.
    #[inline]
    const fn log_instances(&self) -> u32 {
        if self.interleaved {
            self.instances.trailing_zeros()
        } else {
            0
        }
    }

    /// The power of two a slot-local relation position is scaled by.
    #[inline]
    const fn relation_shift(&self) -> u32 {
        if self.scaled {
            self.instances.trailing_zeros()
        } else {
            0
        }
    }

    /// The first relation of an instance within a family.
    #[inline]
    const fn constraint_offset(&self, family: usize, instance: u32) -> u32 {
        if self.scaled {
            self.constraint_base[family] + instance
        } else {
            self.constraint_base[family] + instance * self.strides[family]
        }
    }
}

/// One instance block as the wiring evaluation reads it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockView {
    /// The block's first word.
    pub word_base: usize,
    /// Words one instance occupies.
    pub slots: usize,
    /// Live instances.
    pub instances: usize,
    /// The block's first entry in the slot-key table.
    pub slot_base: usize,
    /// The first relation of the block in each family.
    pub constraint_base: [usize; FAMILIES],
    /// Relations one instance adds to each family.
    pub strides: [usize; FAMILIES],
    /// The base-two logarithm of the instance count when interleaved.
    pub interleaved: Option<usize>,
}

/// Storage the compiled metadata of one segment occupies.
///
/// An instanced layout keeps these counts flat as the instance count grows.
///
/// They are therefore the quantity a scaling measurement should report.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LayoutFootprint {
    /// Stored relation occurrences.
    pub references: usize,
    /// Stored operation-and-shift groups.
    pub keys: usize,
    /// Stored per-slot key spans.
    pub slots: usize,
    /// Stored distinct shift spellings.
    pub shift_sequences: usize,
    /// Stored instance blocks.
    pub blocks: usize,
}

impl LayoutFootprint {
    /// Returns the total of every stored entry.
    #[must_use]
    pub const fn entries(&self) -> usize {
        self.references + self.keys + self.slots + self.shift_sequences + self.blocks
    }

    /// Adds another segment's footprint.
    const fn add(self, other: Self) -> Self {
        Self {
            references: self.references + other.references,
            keys: self.keys + other.keys,
            slots: self.slots + other.slots,
            shift_sequences: self.shift_sequences + other.shift_sequences,
            blocks: self.blocks + other.blocks,
        }
    }
}

/// Metadata compiled independently for one visibility segment.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompiledSegment<W: Word> {
    /// The distinct shift-sequence spellings.
    shifts: Vec<[Shift<W>; 2]>,
    /// The operation and shift groups across all slots.
    keys: Vec<StoredKey>,
    /// The contiguous key span assigned to each slot.
    slot_keys: Vec<Range<u32>>,
    /// The relation consumers grouped by key, in slot-local positions.
    references: Vec<ConstraintReference>,
    /// The calls whose slots repeat across instances, in word order.
    blocks: Vec<InstanceBlock>,
    /// The number of composed words the blocks cover.
    words: usize,
}

impl<W: Word> CompiledSegment<W> {
    /// Returns the number of indexed words.
    #[inline]
    pub const fn len(&self) -> usize {
        self.words
    }

    /// Returns whether the segment has no words.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.words == 0
    }

    /// Returns the storage this segment's metadata occupies.
    #[must_use]
    pub const fn footprint(&self) -> LayoutFootprint {
        LayoutFootprint {
            references: self.references.len(),
            keys: self.keys.len(),
            slots: self.slot_keys.len(),
            shift_sequences: self.shifts.len(),
            blocks: self.blocks.len(),
        }
    }

    /// Returns the resolved keys attached to one word.
    pub fn keys(
        &self,
        word: usize,
    ) -> Option<impl ExactSizeIterator<Item = CompiledKey<'_, W>> + Clone + '_> {
        // Keep storage-local indices inside this segment.
        let word = u32::try_from(word).ok()?;
        let block = self.block_of(word)?;
        let (instance, slot) = block.split(word);
        let range = self.slot_keys.get((block.slot_base + slot) as usize)?;
        let keys = self.keys.get(range.start as usize..range.end as usize)?;

        // Every family shifts by this instance's own share of that family.
        let offsets = [
            block.constraint_offset(0, instance),
            block.constraint_offset(1, instance),
            block.constraint_offset(2, instance),
        ];
        let shift = block.relation_shift() as u8;
        Some(keys.iter().map(move |key| self.resolve(key, offsets, shift)))
    }

    /// The instance blocks in word order.
    pub(crate) fn block_views(&self) -> impl Iterator<Item = BlockView> + '_ {
        self.blocks.iter().map(|block| BlockView {
            word_base: block.word_base as usize,
            slots: block.slots as usize,
            instances: block.instances as usize,
            slot_base: block.slot_base as usize,
            constraint_base: block.constraint_base.map(|base| base as usize),
            strides: block.strides.map(|stride| stride as usize),
            interleaved: (block.interleaved && block.scaled)
                .then(|| block.log_instances() as usize),
        })
    }

    /// The slot-local keys of one stored slot: relation positions unscaled and unshifted.
    pub(crate) fn slot_keys(
        &self,
        slot: usize,
    ) -> impl ExactSizeIterator<Item = CompiledKey<'_, W>> + '_ {
        let range = &self.slot_keys[slot];
        self.keys[range.start as usize..range.end as usize]
            .iter()
            .map(|key| self.resolve(key, [0; FAMILIES], 0))
    }

    /// Returns the dense shift-sequence table.
    #[inline]
    pub fn shift_sequences(&self) -> &[[Shift<W>; 2]] {
        &self.shifts
    }

    /// Returns the block owning one composed word.
    fn block_of(&self, word: u32) -> Option<&InstanceBlock> {
        // Blocks are stored in increasing word order and never overlap.
        let index = self
            .blocks
            .partition_point(|block| block.word_base <= word)
            .checked_sub(1)?;
        let block = &self.blocks[index];
        (word < block.word_base + block.width()).then_some(block)
    }

    /// The wiring flattened for a device, each reference's operand placed by `operand_index`.
    pub(crate) fn shift_layout(
        &self,
        operand_index: impl Fn(ConstraintKind, OperandRole) -> usize,
    ) -> crate::ShiftLayout {
        let blocks = self
            .blocks
            .iter()
            .flat_map(|block| {
                [
                    block.word_base,
                    block.slots,
                    block.instances
                        | (u32::from(block.interleaved) << 31)
                        | (u32::from(block.scaled) << 30),
                    block.slot_base,
                ]
                .into_iter()
                .chain(block.strides)
                .chain(block.constraint_base)
            })
            .collect();
        let slot_keys = self
            .slot_keys
            .iter()
            .flat_map(|range| [range.start, range.end])
            .collect();
        let mut operands = Vec::with_capacity(self.references.len());
        let mut keys = Vec::with_capacity(4 * self.keys.len());
        for key in &self.keys {
            keys.extend([
                u32::from(key.operation.code()),
                key.shift,
                key.references.start,
                key.references.end,
            ]);
        }
        let mut owner = alloc::vec![ConstraintKind::Zero; self.references.len()];
        for key in &self.keys {
            for at in key.references.clone() {
                owner[at as usize] = key.operation;
            }
        }
        for (reference, &kind) in self.references.iter().zip(&owner) {
            let index = operand_index(kind, reference.operand);
            operands.push(u8::try_from(index).expect("an operand position fits a byte"));
        }
        crate::ShiftLayout {
            blocks,
            slot_keys,
            keys,
            operands,
            constraints: self
                .references
                .iter()
                .map(|reference| reference.constraint)
                .collect(),
            shifts: self.shifts.len(),
            sequences: self
                .shifts
                .iter()
                .flat_map(|sequence| {
                    sequence.iter().flat_map(|shift| {
                        [crate::device::shift_kind_code(shift.kind()), shift.amount()]
                    })
                })
                .collect(),
        }
    }

    fn resolve(
        &self,
        key: &StoredKey,
        offsets: [u32; FAMILIES],
        shift: u8,
    ) -> CompiledKey<'_, W> {
        // Construction bounds every dense index and contiguous reference span.
        CompiledKey {
            operation: key.operation,
            shift_index: key.shift,
            shifts: self.shifts[key.shift as usize],
            references: &self.references
                [key.references.start as usize..key.references.end as usize],
            constraint_offset: offsets[key.operation.code() as usize],
            constraint_shift: shift,
        }
    }
}

/// Public and committed shift metadata, kept in separate index spaces.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompiledKeyLayout<W: Word> {
    /// The metadata for verifier-known words.
    public: CompiledSegment<W>,
    /// The metadata for prover-committed words.
    witness: CompiledSegment<W>,
}

impl<W: Word> CompiledKeyLayout<W> {
    /// Compiles every shifted reference in deterministic reduction order.
    ///
    /// - Families use zero, AND, then integer-multiplication order.
    /// - AND roles use left, right, then output order.
    /// - Multiplication roles use left, right, low, then high order.
    /// - Constraint positions ascend within each role.
    /// - Terms retain their supplied order and multiplicity.
    /// - Each word's keys retain the first occurrence of each group.
    pub fn new(system: &ConstraintSystem<W>) -> Result<Self, KeyCompileError> {
        // Each word's references in stream order, then its keys grouped.
        let (public, witness) = tracing::info_span!("place references")
            .in_scope(|| builder::prepare_flat(system))?;
        let _span = tracing::info_span!("group references").entered();
        Ok(Self {
            public: public.compile()?,
            witness: witness.compile()?,
        })
    }

    /// Compiles each component once and repeats it across its instances.
    ///
    /// A call's relations are declared exactly once here, whatever its instance count.
    ///
    /// The stored metadata is therefore the size of the gadgets, not of the statement.
    ///
    /// Compiling the lowered statement instead would give an indistinguishable result.
    ///
    /// That is because the composed addressing is the affine map this one reverses.
    ///
    /// # Errors
    ///
    /// Returns an error when a gadget or the merged layout outgrows the compact key.
    pub fn compose(composition: &Composition<W>) -> Result<Self, KeyCompileError> {
        if composition.placement() == Placement::Interleaved {
            return Self::compose_interleaved(composition);
        }
        let mut public = Vec::with_capacity(composition.calls().len());
        let mut witness = Vec::with_capacity(composition.calls().len());
        let mut constraint_base = [0_u32; FAMILIES];

        for (call, index) in composition.calls().iter().zip(0_usize..) {
            // One compilation of the body serves every instance of the call.
            let compiled = Self::new(call.component().body())?;
            let instances = u32::try_from(call.instances())
                .expect("a laid-out composition bounded every instance count");
            let strides = SegmentPiece::strides(call, index)?;

            public.push(SegmentPiece {
                segment: compiled.public,
                instances,
                strides,
                constraint_base,
                interleaved: false,
                scaled: false,
            });
            witness.push(SegmentPiece {
                segment: compiled.witness,
                instances,
                strides,
                constraint_base,
                interleaved: false,
                scaled: false,
            });

            // The next call's relations start past this call's whole block.
            for (family, stride) in strides.into_iter().enumerate() {
                let kind = ConstraintKind::from_code(family as u8)
                    .expect("relation strides are indexed by assigned family codes");
                let overflow = || KeyCompileError::UnrepresentableComposition { call: index, kind };
                constraint_base[family] = stride
                    .checked_mul(instances)
                    .and_then(|block| constraint_base[family].checked_add(block))
                    .ok_or_else(overflow)?;
            }
        }

        Ok(Self {
            public: CompiledSegment::merge(public, Segment::Public)?,
            witness: CompiledSegment::merge(witness, Segment::Witness)?,
        })
    }

    /// [`Self::compose`] for an interleaved composition: one piece per block, each call's
    /// body compiled once.
    fn compose_interleaved(composition: &Composition<W>) -> Result<Self, KeyCompileError> {
        let compiled = composition
            .calls()
            .iter()
            .map(|call| Self::new(call.component().body()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut public = Vec::with_capacity(composition.blocks().len());
        let mut witness = Vec::with_capacity(composition.blocks().len());
        for block in composition.blocks() {
            let call = &composition.calls()[block.call()];
            let strides = SegmentPiece::strides(call, block.call())?;
            let constraint_base = core::array::from_fn(|family| {
                let kind = ConstraintKind::from_code(family as u8)
                    .expect("relation bases are indexed by assigned family codes");
                block.constraint_base(kind) as u32
            });
            let instances = block.instances() as u32;
            public.push((
                block.public_base(),
                SegmentPiece {
                    segment: compiled[block.call()].public.clone(),
                    instances,
                    strides,
                    constraint_base,
                    interleaved: false,
                    scaled: true,
                },
            ));
            witness.push(SegmentPiece {
                segment: compiled[block.call()].witness.clone(),
                instances,
                strides,
                constraint_base,
                interleaved: true,
                scaled: true,
            });
        }
        public.sort_by_key(|(base, _)| *base);
        let public = public.into_iter().map(|(_, piece)| piece).collect();
        Ok(Self {
            public: CompiledSegment::merge(public, Segment::Public)?,
            witness: CompiledSegment::merge(witness, Segment::Witness)?,
        })
    }

    /// Returns metadata for verifier-known words.
    #[inline]
    pub const fn public(&self) -> &CompiledSegment<W> {
        &self.public
    }

    /// Returns metadata for committed words.
    #[inline]
    pub const fn witness(&self) -> &CompiledSegment<W> {
        &self.witness
    }

    /// Returns the storage both segments' metadata occupies.
    #[must_use]
    pub const fn footprint(&self) -> LayoutFootprint {
        self.public.footprint().add(self.witness.footprint())
    }
}

/// One call's compiled component before the segment blocks are merged.
struct SegmentPiece<W: Word> {
    /// The component body compiled once, in slot-local positions.
    segment: CompiledSegment<W>,
    /// Live instances of the call.
    instances: u32,
    /// Relations one instance adds to each family.
    strides: [u32; FAMILIES],
    /// The first relation position of instance zero.
    constraint_base: [u32; FAMILIES],
    /// Whether the instances' words are interleaved.
    interleaved: bool,
    /// Whether the instances' relations are interleaved.
    scaled: bool,
}

impl<W: Word> SegmentPiece<W> {
    /// Returns the relations one instance of a call adds to each family.
    ///
    /// # Errors
    ///
    /// Returns an error naming the call whose family outgrows the address space.
    fn strides(call: &ComponentCall<W>, index: usize) -> Result<[u32; FAMILIES], KeyCompileError> {
        let mut strides = [0_u32; FAMILIES];
        for (family, count) in call.component().relation_counts().into_iter().enumerate() {
            let kind = ConstraintKind::from_code(family as u8)
                .expect("relation counts are indexed by assigned family codes");
            strides[family] = u32::try_from(count)
                .map_err(|_| KeyCompileError::UnrepresentableComposition { call: index, kind })?;
        }
        Ok(strides)
    }
}

impl<W: Word> CompiledSegment<W> {
    /// Merges one compiled component per call into a single instanced segment.
    ///
    /// Each piece's block starts where the preceding pieces' blocks end.
    ///
    /// # Errors
    ///
    /// Returns an error when the merged segment outgrows the compact index space.
    fn merge(pieces: Vec<SegmentPiece<W>>, segment: Segment) -> Result<Self, KeyCompileError> {
        // Shift spellings are shared, so the merged table is the sorted union.
        let mut codes = pieces
            .iter()
            .flat_map(|piece| {
                piece
                    .segment
                    .shifts
                    .iter()
                    .copied()
                    .map(ShiftSequenceCode::new)
            })
            .collect::<Vec<_>>();
        codes.sort_unstable();
        codes.dedup();
        let shifts = codes
            .iter()
            .copied()
            .map(ShiftSequenceCode::shifts)
            .collect::<Vec<_>>();

        let mut keys = Vec::new();
        let mut slot_keys = Vec::new();
        let mut references = Vec::new();
        let mut blocks = Vec::new();
        let mut words = 0_u32;

        for piece in pieces {
            let bound = |len, component| KeyCompileError::bound(len, segment, component);
            let slot_base = bound(slot_keys.len(), LayoutComponent::WordOffsets)?;
            let key_base = bound(keys.len(), LayoutComponent::Keys)?;
            let reference_base = bound(references.len(), LayoutComponent::References)?;

            // Dense shift indices are rebased onto the merged table.
            let remap = piece
                .segment
                .shifts
                .iter()
                .copied()
                .map(|entry| {
                    codes
                        .binary_search(&ShiftSequenceCode::new(entry))
                        .expect("the merged shift table covers every component spelling")
                        as u32
                })
                .collect::<Vec<_>>();

            references.extend(piece.segment.references.iter().copied());
            keys.extend(piece.segment.keys.iter().map(|key| StoredKey {
                operation: key.operation,
                shift: remap[key.shift as usize],
                references: (key.references.start + reference_base)
                    ..(key.references.end + reference_base),
            }));
            slot_keys.extend(
                piece
                    .segment
                    .slot_keys
                    .iter()
                    .map(|range| (range.start + key_base)..(range.end + key_base)),
            );

            // A call with no words of this segment holds no addressable block.
            let slots = bound(piece.segment.slot_keys.len(), LayoutComponent::WordOffsets)?;
            let too_wide = || KeyCompileError::LayoutTooLarge {
                segment,
                component: LayoutComponent::WordOffsets,
                len: usize::MAX,
            };
            let width = slots.checked_mul(piece.instances).ok_or_else(too_wide)?;
            if width != 0 {
                // The running total is exactly the first word this block owns.
                blocks.push(InstanceBlock {
                    word_base: words,
                    slots,
                    instances: piece.instances,
                    slot_base,
                    strides: piece.strides,
                    constraint_base: piece.constraint_base,
                    interleaved: piece.interleaved,
                    scaled: piece.scaled,
                });
                words = words.checked_add(width).ok_or_else(too_wide)?;
            }
        }

        // Bound every merged endpoint before the segment is handed out.
        KeyCompileError::bound(keys.len(), segment, LayoutComponent::Keys)?;
        KeyCompileError::bound(references.len(), segment, LayoutComponent::References)?;
        KeyCompileError::bound(slot_keys.len(), segment, LayoutComponent::WordOffsets)?;

        Ok(Self {
            shifts,
            keys,
            slot_keys,
            references,
            blocks,
            words: words as usize,
        })
    }
}

/// A bounded part of the compact key layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayoutComponent {
    /// Constraint references.
    References,
    /// Per-word keys.
    Keys,
    /// Word-offset table entries.
    WordOffsets,
}

/// A constraint system no key can represent or prove.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum KeyCompileError {
    /// A segment component cannot be addressed by its compact index.
    #[error("{segment:?} {component:?} has unsupported length {len}")]
    LayoutTooLarge {
        /// The oversized segment.
        segment: Segment,
        /// The oversized component.
        component: LayoutComponent,
        /// The rejected length.
        len: usize,
    },
    /// A call's instances do not fit the compact relation address space.
    #[error("call {call} repeats its {kind:?} relations past the compact address space")]
    UnrepresentableComposition {
        /// The call whose block cannot be addressed.
        call: usize,
        /// The relation family that overflows.
        kind: ConstraintKind,
    },
}

impl KeyCompileError {
    /// Narrows one compiled length to the compact index the layout stores it in.
    ///
    /// # Errors
    ///
    /// Returns the oversized-layout error naming the segment and the component.
    fn bound(len: usize, segment: Segment, component: LayoutComponent) -> Result<u32, Self> {
        u32::try_from(len).map_err(|_| Self::LayoutTooLarge {
            segment,
            component,
            len,
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_word::{
        AndConstraint, IntegerMulConstraint, Operand, ShiftKind, ShiftedValue, ValueIndex, Word32,
        Word64, ZeroConstraint,
    };

    use super::*;

    #[test]
    fn compilation_keeps_segments_and_groups_equal_keys() {
        let public = ValueIndex::public(0).unwrap();
        let witness = ValueIndex::witness(1).unwrap();
        let rotate = Shift::new(ShiftKind::RotateRight, 11).unwrap();
        let public_term = ShiftedValue::<Word64>::single(public, rotate);
        let witness_term = ShiftedValue::<Word64>::plain(witness);
        let left = Operand::new(vec![witness_term, public_term, witness_term]);
        let right = Operand::single(public_term);
        let output = Operand::single(witness_term);
        let system = ConstraintSystem::new(
            1,
            2,
            vec![ZeroConstraint::new(left.clone())],
            vec![AndConstraint::new(left, right, output)],
            vec![],
        )
        .unwrap();

        let layout = CompiledKeyLayout::new(&system).unwrap();
        let public_keys = layout.public().keys(0).unwrap().collect::<Vec<_>>();
        assert_eq!(public_keys.len(), 2);
        assert_eq!(public_keys[0].shifts(), [rotate, Shift::identity()]);
        assert_eq!(
            public_keys[1].references().collect::<Vec<_>>(),
            [
                ConstraintReference {
                    operand: OperandRole::Left,
                    constraint: 0,
                },
                ConstraintReference {
                    operand: OperandRole::Right,
                    constraint: 0,
                },
            ]
        );

        assert_eq!(layout.witness().keys(0).unwrap().len(), 0);
        let witness_keys = layout.witness().keys(1).unwrap().collect::<Vec<_>>();
        assert_eq!(witness_keys.len(), 2);
        assert_eq!(
            witness_keys[0].references().collect::<Vec<_>>(),
            [
                ConstraintReference {
                    operand: OperandRole::Value,
                    constraint: 0,
                },
                ConstraintReference {
                    operand: OperandRole::Value,
                    constraint: 0,
                },
            ]
        );
    }

    #[test]
    fn operation_and_operand_order_matches_the_reduction_order() {
        let index = ValueIndex::witness(0).unwrap();
        let term = ShiftedValue::<Word32>::plain(index);
        let value = Operand::single(term);
        let system = ConstraintSystem::new(
            0,
            1,
            vec![ZeroConstraint::new(value.clone())],
            vec![AndConstraint::new(
                value.clone(),
                value.clone(),
                value.clone(),
            )],
            vec![IntegerMulConstraint::new(
                value.clone(),
                value.clone(),
                value.clone(),
                value,
            )],
        )
        .unwrap();

        let layout = CompiledKeyLayout::new(&system).unwrap();
        let keys = layout.witness().keys(0).unwrap().collect::<Vec<_>>();
        assert_eq!(
            keys.iter().map(|key| key.operation()).collect::<Vec<_>>(),
            [
                ConstraintKind::Zero,
                ConstraintKind::And,
                ConstraintKind::IntegerMul,
            ]
        );
        assert_eq!(
            keys[1].references().collect::<Vec<_>>(),
            [
                ConstraintReference {
                    operand: OperandRole::Left,
                    constraint: 0,
                },
                ConstraintReference {
                    operand: OperandRole::Right,
                    constraint: 0,
                },
                ConstraintReference {
                    operand: OperandRole::Output,
                    constraint: 0,
                },
            ]
        );
    }

    #[test]
    fn dense_shifts_keep_word64_lane_semantics_distinct() {
        let index = ValueIndex::witness(0).unwrap();
        let full = Shift::new(ShiftKind::LogicalRight, 5).unwrap();
        let lanes = Shift::new(ShiftKind::Lane32LogicalRight, 5).unwrap();
        let system = ConstraintSystem::new(
            0,
            1,
            vec![ZeroConstraint::new(Operand::new(vec![
                ShiftedValue::<Word64>::single(index, full),
                ShiftedValue::<Word64>::single(index, lanes),
            ]))],
            vec![],
            vec![],
        )
        .unwrap();

        let layout = CompiledKeyLayout::new(&system).unwrap();
        let keys = layout.witness().keys(0).unwrap().collect::<Vec<_>>();
        assert_eq!(keys.len(), 2);
        assert_ne!(keys[0].shifts(), keys[1].shifts());
        assert_ne!(keys[0].shift_index(), keys[1].shift_index());
    }

    #[test]
    fn two_shift_sequences_round_trip_through_the_dense_encoding() {
        let index = ValueIndex::witness(0).unwrap();
        let inner = Shift::new(ShiftKind::LogicalLeft, 9).unwrap();
        let outer = Shift::new(ShiftKind::RotateRight, 17).unwrap();
        let pair = ShiftedValue::<Word64>::pair(index, inner, outer).unwrap();
        let system = ConstraintSystem::new(
            0,
            1,
            vec![ZeroConstraint::new(Operand::single(pair))],
            vec![],
            vec![],
        )
        .unwrap();

        let layout = CompiledKeyLayout::new(&system).unwrap();
        let key = layout.witness().keys(0).unwrap().next().unwrap();
        assert_eq!(key.shifts(), [inner, outer]);
        assert_eq!(
            layout.witness().shift_sequences()[key.shift_index() as usize],
            [inner, outer]
        );
    }

    #[test]
    fn resolved_keys_cannot_cross_segment_boundaries() {
        // Public and committed words deliberately use different movements at the same position.
        let public = ValueIndex::public(0).unwrap();
        let witness = ValueIndex::witness(0).unwrap();
        let rotate = Shift::new(ShiftKind::RotateRight, 3).unwrap();
        let left = Shift::new(ShiftKind::LogicalLeft, 5).unwrap();
        let system = ConstraintSystem::new(
            1,
            1,
            vec![ZeroConstraint::new(Operand::new(vec![
                ShiftedValue::<Word64>::single(public, rotate),
                ShiftedValue::<Word64>::single(witness, left),
            ]))],
            vec![],
            vec![],
        )
        .unwrap();

        // Each iterator resolves indices only through the segment that owns them.
        let public_key = layout_key(&system, Segment::Public);
        let witness_key = layout_key(&system, Segment::Witness);
        assert_eq!(public_key, [rotate, Shift::identity()]);
        assert_eq!(witness_key, [left, Shift::identity()]);
    }

    fn layout_key(system: &ConstraintSystem<Word64>, segment: Segment) -> [Shift<Word64>; 2] {
        // Select the independent table whose key must be resolved.
        let layout = CompiledKeyLayout::new(system).unwrap();
        let compiled = match segment {
            Segment::Public => layout.public(),
            Segment::Witness => layout.witness(),
        };
        compiled.keys(0).unwrap().next().unwrap().shifts()
    }

    #[test]
    fn more_than_u16_shift_sequences_compile() {
        // Enumerate every checked movement once before forming irreducible pairs.
        let index = ValueIndex::witness(0).unwrap();
        let mut shifts = vec![Shift::<Word64>::identity()];
        for kind in [
            ShiftKind::LogicalLeft,
            ShiftKind::LogicalRight,
            ShiftKind::ArithmeticRight,
            ShiftKind::RotateRight,
            ShiftKind::Lane32LogicalLeft,
            ShiftKind::Lane32LogicalRight,
            ShiftKind::Lane32ArithmeticRight,
            ShiftKind::Lane32RotateRight,
        ] {
            let width = if kind.is_lane32() { 32 } else { 64 };
            for amount in 1..width {
                shifts.push(Shift::new(kind, amount).unwrap());
            }
        }

        // Fixture state: 65,537 distinct spellings exceed the former 16-bit index space.
        let target = u16::MAX as usize + 2;
        let mut terms = shifts
            .iter()
            .copied()
            .map(|shift| ShiftedValue::single(index, shift))
            .collect::<Vec<_>>();
        'outer: for &inner in &shifts {
            for &outer in &shifts {
                if let Ok(term) = ShiftedValue::pair(index, inner, outer) {
                    terms.push(term);
                    if terms.len() == target {
                        break 'outer;
                    }
                }
            }
        }
        assert_eq!(terms.len(), target);

        // A 32-bit dense index preserves every valid shift spelling.
        let system = ConstraintSystem::new(
            0,
            1,
            vec![ZeroConstraint::new(Operand::new(terms))],
            vec![],
            vec![],
        )
        .unwrap();
        let layout = CompiledKeyLayout::new(&system).unwrap();
        let keys = layout.witness().keys(0).unwrap();
        assert_eq!(keys.len(), target);
        assert_eq!(layout.witness().shift_sequences().len(), target);
        assert!(keys.map(|key| key.shift_index()).max().unwrap() > u32::from(u16::MAX));
    }
}
