//! Exact construction of per-segment key storage.

use alloc::vec;
use alloc::vec::Vec;

use core::ops::Range;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use p3_maybe_rayon::prelude::*;
use p3_word::{ConstraintKind, ConstraintSystem, OperandRole, Segment, ShiftedValue, Word};

use super::code::{KEY_CODES, KeyCode, ShiftSequenceCode};
use super::{
    CompiledSegment, ConstraintReference, FAMILIES, InstanceBlock, KeyCompileError,
    LayoutComponent, StoredKey,
};

/// The term stream's eight operand columns in [`ConstraintSystem::terms`] order: each family,
/// then each of its operand roles.
const COLUMNS: [(ConstraintKind, OperandRole); 8] = [
    (ConstraintKind::Zero, OperandRole::Value),
    (ConstraintKind::And, OperandRole::Left),
    (ConstraintKind::And, OperandRole::Right),
    (ConstraintKind::And, OperandRole::Output),
    (ConstraintKind::IntegerMul, OperandRole::Left),
    (ConstraintKind::IntegerMul, OperandRole::Right),
    (ConstraintKind::IntegerMul, OperandRole::Low),
    (ConstraintKind::IntegerMul, OperandRole::High),
];

/// The terms of operand column `column` of relation `constraint`.
fn column_terms<W: Word>(
    system: &ConstraintSystem<W>,
    column: usize,
    constraint: usize,
) -> &[ShiftedValue<W>] {
    match column {
        0 => system.zero_constraints()[constraint].value().terms(),
        1 => system.and_constraints()[constraint].left().terms(),
        2 => system.and_constraints()[constraint].right().terms(),
        3 => system.and_constraints()[constraint].output().terms(),
        4 => system.integer_mul_constraints()[constraint].left().terms(),
        5 => system.integer_mul_constraints()[constraint].right().terms(),
        6 => system.integer_mul_constraints()[constraint].low().terms(),
        _ => system.integer_mul_constraints()[constraint].high().terms(),
    }
}

/// How many relations operand column `column` has.
fn column_len<W: Word>(system: &ConstraintSystem<W>, column: usize) -> usize {
    match column {
        0 => system.zero_constraints().len(),
        1..=3 => system.and_constraints().len(),
        _ => system.integer_mul_constraints().len(),
    }
}

/// A run of the term stream: relations `constraints` of operand column `column`.
struct Run {
    column: usize,
    constraints: Range<usize>,
}

impl Run {
    /// Calls `visit` with each of the run's references and its word, the public words first
    /// and the committed words after them, in stream order.
    fn each<W: Word>(
        &self,
        system: &ConstraintSystem<W>,
        mut visit: impl FnMut(usize, Segment, PackedReference),
    ) {
        let (kind, role) = COLUMNS[self.column];
        let public = system.public_len();
        for constraint in self.constraints.clone() {
            for term in column_terms(system, self.column, constraint) {
                let index = term.index();
                let segment = index.segment();
                let position = index.position() as usize;
                let word = match segment {
                    Segment::Public => position,
                    Segment::Witness => public + position,
                };
                visit(
                    word,
                    segment,
                    PackedReference {
                        key_code: KeyCode::new(kind, term.shifts()),
                        constraint: ConstraintReference {
                            operand: role,
                            constraint: constraint as u32,
                        },
                    },
                );
            }
        }
    }
}

/// Both segments' reference tables of a flat statement, each word's references in stream
/// order, built in parallel: the stream is cut into runs, every run counts its references per word bucket,
/// writes them into its share of each bucket, and each bucket then sorts its references by
/// word, keeping their order.
pub(super) fn prepare_flat<W: Word>(
    system: &ConstraintSystem<W>,
) -> Result<(PreparedSegment, PreparedSegment), KeyCompileError> {
    let (public_len, witness_len) = (system.public_len(), system.witness_len());
    let words = public_len + witness_len;
    let too_large = |segment, len| KeyCompileError::LayoutTooLarge {
        segment,
        component: LayoutComponent::References,
        len,
    };

    // The stream cut into runs of about equal relation counts within each column.
    let slots: usize = (0..COLUMNS.len()).map(|c| column_len(system, c)).sum();
    let target = slots.div_ceil(8 * current_num_threads().max(1)).max(1);
    let runs: Vec<Run> = (0..COLUMNS.len())
        .flat_map(|column| {
            let len = column_len(system, column);
            (0..len)
                .step_by(target)
                .map(move |first| Run { column, constraints: first..(first + target).min(len) })
        })
        .collect();
    let shift = (usize::BITS - words.max(1).leading_zeros()).saturating_sub(12);
    let buckets = (words >> shift) + 1;

    // Counts per word and per run and bucket, and the key codes each segment uses.
    let counts: Vec<AtomicUsize> = (0..=words).map(|_| AtomicUsize::new(0)).collect();
    let seen: [Vec<AtomicBool>; 2] =
        core::array::from_fn(|_| (0..KEY_CODES).map(|_| AtomicBool::new(false)).collect());
    let run_counts: Vec<Vec<u32>> = runs
        .par_iter()
        .map(|run| {
            let mut local = vec![0u32; buckets];
            run.each(system, |word, segment, reference| {
                counts[word + 1].fetch_add(1, Ordering::Relaxed);
                let flag = &seen[usize::from(segment == Segment::Witness)][reference.key_code.index()];
                if !flag.load(Ordering::Relaxed) {
                    flag.store(true, Ordering::Relaxed);
                }
                local[word >> shift] += 1;
            });
            local
        })
        .collect();
    let mut offsets: Vec<usize> = counts.into_iter().map(AtomicUsize::into_inner).collect();
    for word in 0..words {
        offsets[word + 1] += offsets[word];
    }
    let total = offsets[words];
    if u32::try_from(total).is_err() {
        return Err(too_large(Segment::Witness, total));
    }

    // Every run's share of every bucket, in stream order within the bucket.
    let mut references = vec![PackedReference::default(); total];
    let mut word_of = vec![0u32; total];
    let bucket_range = |bucket: usize| {
        offsets[(bucket << shift).min(words)]..offsets[((bucket + 1) << shift).min(words)]
    };
    {
        let mut shares: Vec<Vec<(&mut [PackedReference], &mut [u32])>> =
            runs.iter().map(|_| Vec::with_capacity(buckets)).collect();
        let (mut references_left, mut words_left) = (&mut references[..], &mut word_of[..]);
        for bucket in 0..buckets {
            for (run, share) in run_counts.iter().zip(&mut shares) {
                let len = run[bucket] as usize;
                let (head, tail) = references_left.split_at_mut(len);
                references_left = tail;
                let (words_head, words_tail) = words_left.split_at_mut(len);
                words_left = words_tail;
                share.push((head, words_head));
            }
        }
        debug_assert!(references_left.is_empty() && bucket_range(buckets - 1).end == total);
        runs.par_iter().zip(shares.into_par_iter()).for_each(|(run, mut share)| {
            let mut cursors = vec![0usize; buckets];
            run.each(system, |word, _, reference| {
                let bucket = word >> shift;
                let (references, words) = &mut share[bucket];
                references[cursors[bucket]] = reference;
                words[cursors[bucket]] = word as u32;
                cursors[bucket] += 1;
            });
        });
    }
    drop(run_counts);

    // Each bucket's references sorted by word, stably.
    {
        let mut regions = Vec::with_capacity(buckets);
        let (mut references_left, mut words_left) = (&mut references[..], &mut word_of[..]);
        for bucket in 0..buckets {
            let range = bucket_range(bucket);
            let (head, tail) = references_left.split_at_mut(range.len());
            references_left = tail;
            let (words_head, words_tail) = words_left.split_at_mut(range.len());
            words_left = words_tail;
            regions.push((bucket, range.start, head, words_head));
        }
        regions.into_par_iter().for_each(|(bucket, start, references, words)| {
            let first = bucket << shift;
            let last = ((bucket + 1) << shift).min(offsets.len() - 1);
            if first >= last {
                return;
            }
            let mut cursors: Vec<usize> =
                offsets[first..last].iter().map(|&offset| offset - start).collect();
            let staged: Vec<(PackedReference, u32)> =
                references.iter().copied().zip(words.iter().copied()).collect();
            for (reference, word) in staged {
                let cursor = &mut cursors[word as usize - first];
                references[*cursor] = reference;
                *cursor += 1;
            }
        });
    }
    drop(word_of);

    // The two segments' tables, the committed words' counted from their own first word.
    let witness_references = references.split_off(offsets[public_len]);
    let base = offsets[public_len];
    let witness_offsets: Vec<usize> = offsets[public_len..].iter().map(|&o| o - base).collect();
    offsets.truncate(public_len + 1);
    let codes = |seen: &[AtomicBool]| -> Vec<KeyCode> {
        (0..KEY_CODES)
            .filter(|&code| seen[code].load(Ordering::Relaxed))
            .map(KeyCode::from_index)
            .collect()
    };
    let [public_seen, witness_seen] = &seen;
    Ok((
        PreparedSegment {
            segment: Segment::Public,
            offsets,
            references,
            key_codes: codes(public_seen),
        },
        PreparedSegment {
            segment: Segment::Witness,
            offsets: witness_offsets,
            references: witness_references,
            key_codes: codes(witness_seen),
        },
    ))
}

/// One visibility segment's references, each word's in stream order.
pub(super) struct PreparedSegment {
    /// The visibility class being compiled.
    segment: Segment,
    /// The start offset of each word and one trailing total.
    offsets: Vec<usize>,
    /// The references placed in word-major order.
    references: Vec<PackedReference>,
    /// The distinct key codes, ascending.
    key_codes: Vec<KeyCode>,
}

impl PreparedSegment {
    /// Compacts the populated reference table into immutable per-word keys.
    ///
    /// Each word's keys follow the first occurrence of each key among its references, and each
    /// key's references keep their order; the words are compacted in parallel, each into the
    /// span the counts before it leave.
    pub(super) fn compile<W: Word>(self) -> Result<CompiledSegment<W>, KeyCompileError> {
        // Dense key and shift tables make repeated spellings share one index.
        let mut sequence_codes = self
            .key_codes
            .iter()
            .copied()
            .map(KeyCode::sequence)
            .collect::<Vec<_>>();
        sequence_codes.sort_unstable();
        sequence_codes.dedup();
        let shifts = sequence_codes
            .iter()
            .copied()
            .map(ShiftSequenceCode::shifts)
            .collect::<Vec<_>>();
        let mut id_of = vec![u32::MAX; KEY_CODES];
        for (id, code) in self.key_codes.iter().enumerate() {
            id_of[code.index()] = id as u32;
        }
        let key_shift = self
            .key_codes
            .iter()
            .map(|code| {
                let shift = sequence_codes
                    .binary_search(&code.sequence())
                    .expect("the dense shift table covers every compact key");
                u32::try_from(shift).expect("two nine-bit shift codes fit in a 32-bit dense index")
            })
            .collect::<Vec<_>>();

        let words = self.offsets.len().saturating_sub(1);
        let span = |word: usize| &self.references[self.offsets[word]..self.offsets[word + 1]];
        let groups = WordGroups { id_of: &id_of, keys: self.key_codes.len() };

        // Each word's distinct keys, then where its keys start.
        let distinct = (0..words)
            .into_par_iter()
            .map_init(|| groups.scratch(), |scratch, word| groups.count(scratch, span(word)))
            .collect::<Vec<u32>>();
        let mut key_starts = Vec::with_capacity(words + 1);
        let mut total = 0usize;
        key_starts.push(0u32);
        for &count in &distinct {
            total += count as usize;
            key_starts.push(u32::try_from(total).map_err(|_| KeyCompileError::LayoutTooLarge {
                segment: self.segment,
                component: LayoutComponent::Keys,
                len: total,
            })?);
        }
        drop(distinct);

        // Every word's keys and references land in the spans the prefix sums give it.
        let placeholder = StoredKey {
            operation: ConstraintKind::Zero,
            shift: 0,
            references: 0..0,
        };
        let mut keys = vec![placeholder; total];
        let mut flat_references = vec![
            ConstraintReference {
                operand: OperandRole::Value,
                constraint: 0,
            };
            self.references.len()
        ];
        let chunk = words.div_ceil(4 * current_num_threads().max(1)).max(1);
        let mut pieces = Vec::with_capacity(words.div_ceil(chunk));
        let (mut keys_left, mut references_left) = (&mut keys[..], &mut flat_references[..]);
        for first in (0..words).step_by(chunk) {
            let last = (first + chunk).min(words);
            let (piece_keys, rest) = keys_left
                .split_at_mut((key_starts[last] - key_starts[first]) as usize);
            keys_left = rest;
            let (piece_references, rest) = references_left
                .split_at_mut(self.offsets[last] - self.offsets[first]);
            references_left = rest;
            pieces.push((first..last, piece_keys, piece_references));
        }
        pieces
            .into_par_iter()
            .for_each_init(
                || groups.scratch(),
                |scratch, (range, piece_keys, piece_references)| {
                    let (key_base, reference_base) =
                        (key_starts[range.start] as usize, self.offsets[range.start]);
                    for word in range {
                        groups.place(
                            scratch,
                            span(word),
                            self.offsets[word] as u32,
                            &self.key_codes,
                            &key_shift,
                            &mut piece_keys[key_starts[word] as usize - key_base..],
                            &mut piece_references[self.offsets[word] - reference_base..],
                        );
                    }
                },
            );
        let word_keys = key_starts.windows(2).map(|w| w[0]..w[1]).collect::<Vec<_>>();

        // A flat statement is one block of one instance: slots are its words.
        let slots =
            u32::try_from(word_keys.len()).map_err(|_| KeyCompileError::LayoutTooLarge {
                segment: self.segment,
                component: LayoutComponent::WordOffsets,
                len: word_keys.len(),
            })?;
        let words = word_keys.len();
        let blocks = (slots != 0)
            .then_some(InstanceBlock {
                word_base: 0,
                slots,
                instances: 1,
                slot_base: 0,
                strides: [0; FAMILIES],
                constraint_base: [0; FAMILIES],
                interleaved: false,
                scaled: false,
            })
            .into_iter()
            .collect();

        Ok(CompiledSegment {
            shifts,
            keys,
            slot_keys: word_keys,
            references: flat_references,
            blocks,
            words,
        })
    }
}

/// Groups one word's references by key, keys in order of first occurrence.
#[derive(Clone, Copy)]
struct WordGroups<'a> {
    /// Each key code's dense index.
    id_of: &'a [u32],
    /// How many distinct keys there are.
    keys: usize,
}

/// One thread's working tables for [`WordGroups`]: each key's group in the current word, and
/// each group's key and next free reference.
struct GroupScratch {
    slot_of: Vec<u32>,
    groups: Vec<(u32, u32)>,
}

impl WordGroups<'_> {
    fn scratch(&self) -> GroupScratch {
        GroupScratch { slot_of: vec![u32::MAX; self.keys], groups: Vec::new() }
    }

    /// Fills `scratch.groups` with `references`' keys in order of first occurrence, each with
    /// its reference count.
    fn group(&self, scratch: &mut GroupScratch, references: &[PackedReference]) {
        scratch.groups.clear();
        for reference in references {
            let id = self.id_of[reference.key_code.index()];
            let slot = &mut scratch.slot_of[id as usize];
            if *slot == u32::MAX {
                *slot = scratch.groups.len() as u32;
                scratch.groups.push((id, 1));
            } else {
                scratch.groups[*slot as usize].1 += 1;
            }
        }
        for &(id, _) in &scratch.groups {
            scratch.slot_of[id as usize] = u32::MAX;
        }
    }

    /// How many distinct keys `references` has.
    fn count(&self, scratch: &mut GroupScratch, references: &[PackedReference]) -> u32 {
        self.group(scratch, references);
        scratch.groups.len() as u32
    }

    /// Writes the keys of `references`, whose first reference lands at `base` in the whole
    /// table, into `keys`, and their references grouped by key into `out`.
    #[allow(clippy::too_many_arguments)]
    fn place(
        &self,
        scratch: &mut GroupScratch,
        references: &[PackedReference],
        base: u32,
        key_codes: &[KeyCode],
        key_shift: &[u32],
        keys: &mut [StoredKey],
        out: &mut [ConstraintReference],
    ) {
        self.group(scratch, references);
        let mut start = 0u32;
        for (slot, group) in scratch.groups.iter_mut().enumerate() {
            let (id, count) = *group;
            let code = key_codes[id as usize];
            keys[slot] = StoredKey {
                operation: code.operation(),
                shift: key_shift[id as usize],
                references: base + start..base + start + count,
            };
            group.1 = start;
            start += count;
            scratch.slot_of[id as usize] = slot as u32;
        }
        for reference in references {
            let id = self.id_of[reference.key_code.index()];
            let group = &mut scratch.groups[scratch.slot_of[id as usize] as usize];
            out[group.1 as usize] = reference.constraint;
            group.1 += 1;
        }
        for &(id, _) in &scratch.groups {
            scratch.slot_of[id as usize] = u32::MAX;
        }
    }
}

/// One compact reference before equal-key groups are flattened.
#[derive(Clone, Copy)]
struct PackedReference {
    /// The encoded operation and shift sequence.
    key_code: KeyCode,
    /// The relation consumer of the shifted word.
    constraint: ConstraintReference,
}

impl Default for PackedReference {
    fn default() -> Self {
        // Every placeholder is overwritten during the exact placement pass.
        Self {
            key_code: KeyCode::default(),
            constraint: ConstraintReference {
                operand: OperandRole::Value,
                constraint: 0,
            },
        }
    }
}
