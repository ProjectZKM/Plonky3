//! Sparse evaluation of the public shift wiring.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::Gf2;
use p3_field::{Algebra, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
#[cfg(test)]
use p3_word::ConstraintSystem;
use p3_word::{ConstraintKind, OperandRole, Shift, Word};

use super::ShiftClaim;
use super::polynomial::SumOfProducts;
use super::transcript::BatchWeights;
use crate::keys::BlockView;
use crate::{CompiledKey, CompiledKeyLayout, CompiledSegment, PackedWitness, PackedWord};

/// Equality and batching tables shared by both proving phases.
pub(super) struct PreparedWeights<F> {
    /// One equality weight per padded constraint row.
    constraint: Constraints<F>,
    /// The statement point the constraint weights are the equality table of.
    point: Vec<F>,
    /// One equality weight per relation-family slot.
    operation: Vec<F>,
    /// One equality weight per operand-position slot.
    operand: Vec<F>,
}

/// The equality weights of the constraint rows: a table of every row, or the point alone, each
/// row's weight then evaluated when it is read.
enum Constraints<F> {
    Table(Vec<F>),
    Point(Vec<F>),
}

impl<F: Field> Constraints<F> {
    /// The equality weight of `row`, its leading bit matched to the point's first coordinate.
    fn weight(&self, row: usize) -> F {
        match self {
            Self::Table(table) => table[row],
            Self::Point(point) => {
                let n = point.len();
                point
                    .iter()
                    .enumerate()
                    .map(|(i, &r)| {
                        if (row >> (n - 1 - i)) & 1 == 1 {
                            r
                        } else {
                            F::ONE - r
                        }
                    })
                    .product()
            }
        }
    }
}

impl<F: Field> PreparedWeights<F> {
    /// Expands the statement point and retains the transcript batching tables.
    pub(super) fn new(claim: &ShiftClaim<F>, batch: BatchWeights<F>) -> Self {
        // Every compiled reference reads one entry from each table.
        Self {
            constraint: Constraints::Table(
                Point::new(claim.constraint_point()).equality_weights_msb(),
            ),
            point: claim.constraint_point().to_vec(),
            operation: batch.operation,
            operand: batch.operand,
        }
    }

    /// As [`Self::new`], each row's weight evaluated only when it is read: for a verifier,
    /// which reads the rows of the public words alone.
    pub(super) fn sparse(claim: &ShiftClaim<F>, batch: BatchWeights<F>) -> Self {
        Self {
            constraint: Constraints::Point(claim.constraint_point().to_vec()),
            point: claim.constraint_point().to_vec(),
            operation: batch.operation,
            operand: batch.operand,
        }
    }

    /// The tables of the statement point `constraint_point` and the batching weights.
    pub(super) fn from_parts(constraint_point: &[F], operation: Vec<F>, operand: Vec<F>) -> Self {
        Self {
            constraint: Constraints::Table(Point::new(constraint_point).equality_weights_msb()),
            point: constraint_point.to_vec(),
            operation,
            operand,
        }
    }

    /// As [`Self::from_parts`], each row's weight evaluated only when it is read.
    pub(super) fn sparse_from_parts(
        constraint_point: &[F],
        operation: Vec<F>,
        operand: Vec<F>,
    ) -> Self {
        Self {
            constraint: Constraints::Point(constraint_point.to_vec()),
            point: constraint_point.to_vec(),
            operation,
            operand,
        }
    }

    /// Sums all relation occurrences grouped under one compiled key.
    fn key<W: Word>(&self, key: &CompiledKey<'_, W>) -> F {
        // The family weight is shared by every occurrence under one key.
        let operation = self.operation[key.operation().code() as usize];

        // Repeated terms add in the challenge field and cancel in characteristic two.
        let occurrences = key
            .references()
            .map(|reference| {
                // The compiled role is translated into its family-local operand position.
                let operand = operand_index(key.operation(), reference.operand());
                self.operand[operand] * self.constraint.weight(reference.constraint() as usize)
            })
            .sum::<F>();
        operation * occurrences
    }

    /// The family weights, then the operand weights.
    pub(super) fn batching(&self) -> (&[F], &[F]) {
        (&self.operation, &self.operand)
    }

    /// Batches the supplied operand claims with these transcript weights.
    pub(super) fn claim(&self, claim: &ShiftClaim<F>) -> F {
        // Centralizing the order prevents claim batching and sparse wiring from diverging.
        claim.batched(&self.operation, &self.operand)
    }
}

/// Returns a role's position in the operand batching cube.
pub(super) fn operand_index(kind: ConstraintKind, role: OperandRole) -> usize {
    // Each family uses the semantic order exposed by its relation type.
    match (kind, role) {
        (ConstraintKind::Zero, OperandRole::Value)
        | (ConstraintKind::And, OperandRole::Left)
        | (ConstraintKind::IntegerMul, OperandRole::Left) => 0,
        (ConstraintKind::And, OperandRole::Right)
        | (ConstraintKind::IntegerMul, OperandRole::Right) => 1,
        (ConstraintKind::And, OperandRole::Output)
        | (ConstraintKind::IntegerMul, OperandRole::Low) => 2,
        (ConstraintKind::IntegerMul, OperandRole::High) => 3,
        _ => unreachable!("a checked relation never pairs with a foreign operand role"),
    }
}

/// Evaluates one word's bit multilinear at a point.
pub(super) fn evaluate_word<W: Word, F: Field>(word: W, weights: &[F]) -> F {
    // Set bits select the equality weights that survive the Boolean word.
    let bits = word.to_u64();
    (0..W::BITS as usize)
        .filter(|&bit| (bits >> bit) & 1 == 1)
        .map(|bit| weights[bit])
        .sum()
}

/// Returns one subset-sum table per byte of a word, so a word evaluates in one lookup per byte.
///
/// ```text
///     tables[j][b] = sum_{t in b} weights[8 * j + t]
/// ```
pub(crate) fn byte_tables<F: Field>(weights: &[F]) -> Vec<[F; 256]> {
    weights
        .chunks(8)
        .map(|weights| {
            // Each subset extends the one without its lowest bit.
            let mut table = [F::ZERO; 256];
            for byte in 1..256_usize {
                let low = byte.trailing_zeros() as usize;
                table[byte] = table[byte & (byte - 1)] + weights[low];
            }
            table
        })
        .collect()
}

/// Evaluates one word's bit multilinear through its byte tables.
pub(crate) fn evaluate_bytes<F: Field>(bits: u64, tables: &[[F; 256]]) -> F {
    tables
        .iter()
        .enumerate()
        .map(|(byte, table)| table[((bits >> (8 * byte)) & 0xff) as usize])
        .sum()
}

/// Words one task scatters before its rows join the shared sum.
const SCATTER_CHUNK: usize = 1 << 14;

/// Applies the transpose of one fixed shift to a bit-evaluation table.
fn shift_transpose<W: Word, F: Field>(shift: Shift<W>, output: &[F]) -> Vec<F> {
    // An identity preserves the table and needs no basis walk.
    if shift.is_identity() {
        return output.to_vec();
    }

    // Column `j` records where input basis bit `j` reaches after this shift.
    // Every basis image of one shift has at most one bit, except that sign extension fans the top bit out.
    // The set bits across all images therefore total at most two word widths, so walking only set bits is linear.
    (0..W::BITS as usize)
        .map(|input| {
            let basis = W::from_low_bits(1_u64 << input);
            let mut shifted = shift.apply(basis).to_u64();
            let mut sum = F::ZERO;
            while shifted != 0 {
                sum += output[shifted.trailing_zeros() as usize];
                shifted &= shifted - 1;
            }
            sum
        })
        .collect()
}

/// Builds the factorized transpose action of one two-slot shift sequence.
fn shift_weights<W: Word, F: Field>(shifts: [Shift<W>; 2], output: &[F]) -> Vec<F> {
    // Transposing a composed map applies the outer slot first and the inner one second.
    //
    // Keeping the two slots apart avoids materializing a dense composed operator.
    //
    // It also leaves their protocol order visible.
    let [inner, outer] = shifts;
    let intermediate = shift_transpose(outer, output);
    shift_transpose(inner, &intermediate)
}

/// Returns the public-segment share of the batched operand claim.
pub(super) fn public_contribution<W, F>(
    segment: &CompiledSegment<W>,
    words: &[W],
    first: usize,
    prepared: &PreparedWeights<F>,
    bit_point: &[F],
) -> F
where
    W: Word,
    F: Field + Algebra<Gf2>,
{
    // One equality table serves every shifted public word.
    let output = Point::new(bit_point).equality_weights_msb();
    let mut contribution = F::ZERO;

    for (word_index, &word) in words.iter().enumerate().map(|(i, w)| (first + i, w)) {
        // Every key fixes one shift sequence and one relation family.
        let keys = segment
            .keys(word_index)
            .expect("the checked public segment contains this word");
        for key in keys {
            let coefficient = prepared.key(&key);
            if coefficient == F::ZERO {
                continue;
            }

            // Applying the fixed shifts before evaluation is the public reference path.
            let [inner, outer] = key.shifts();
            let shifted = outer.apply(inner.apply(word));
            contribution += coefficient * evaluate_word(shifted, &output);
        }
    }
    contribution
}

/// Per public word from `0` to `count`, the weight of each of its bits in [`public_contribution`]:
/// the contribution of words `0..count` is the sum of the weights of their set bits.
pub(super) fn public_bit_weights<W, F>(
    segment: &CompiledSegment<W>,
    count: usize,
    prepared: &PreparedWeights<F>,
    bit_point: &[F],
) -> Vec<Vec<F>>
where
    W: Word,
    F: Field + Algebra<Gf2>,
{
    // Every shift is linear over GF(2), so a word's share is the sum of its bits' shares.
    let output = Point::new(bit_point).equality_weights_msb();
    (0..count)
        .map(|word_index| {
            let mut weights = vec![F::ZERO; W::BITS as usize];
            let keys = segment
                .keys(word_index)
                .expect("the checked public segment contains this word");
            for key in keys {
                let coefficient = prepared.key(&key);
                if coefficient == F::ZERO {
                    continue;
                }
                let [inner, outer] = key.shifts();
                for (bit, weight) in weights.iter_mut().enumerate() {
                    let shifted = outer.apply(inner.apply(W::from_low_bits(1 << bit)));
                    *weight += coefficient * evaluate_word(shifted, &output);
                }
            }
            weights
        })
        .collect()
}

/// Constructs the sparse bit-index sumcheck and its shift rows.
pub(super) fn bit_prover<W, F>(
    layout: &CompiledKeyLayout<W>,
    values: &PackedWitness<W>,
    prepared: &PreparedWeights<F>,
    bit_point: &[F],
) -> SumOfProducts<F>
where
    W: PackedWord,
    F: Field + Algebra<Gf2>,
{
    // One row is retained per shift spelling used by the committed segment.
    let segment = layout.witness();
    let sequences = segment.shift_sequences();
    let width = W::BITS as usize;

    // Each task scatters whole chunks into its own rows, and the rows add at the end.
    let rows = values
        .witness()
        .par_chunks(SCATTER_CHUNK)
        .enumerate()
        .par_fold_reduce(
            || F::zero_vec(sequences.len() * width),
            |mut rows, (chunk, words)| {
                for (offset, &packed) in words.iter().enumerate() {
                    // Source words remain bit-packed while coefficients are scattered into rows.
                    let bits = W::unpack(packed).to_u64();
                    let keys = segment
                        .keys(chunk * SCATTER_CHUNK + offset)
                        .expect("the checked witness segment contains this word");
                    for key in keys {
                        let coefficient = prepared.key(&key);
                        if coefficient == F::ZERO {
                            continue;
                        }
                        let row = &mut rows[key.shift_index() as usize * width..][..width];
                        let mut remaining = bits;
                        while remaining != 0 {
                            // Visit only set lanes; random 64-bit words halve the scatter work.
                            let bit = remaining.trailing_zeros() as usize;
                            row[bit] += coefficient;
                            remaining &= remaining - 1;
                        }
                    }
                }
                rows
            },
            |mut rows, other| {
                rows.iter_mut()
                    .zip(other)
                    .for_each(|(row, value)| *row += value);
                rows
            },
        );
    bit_pairs(layout, &rows, bit_point)
}

/// The bit phase over the witness rows `rows`, `W::BITS` per shift sequence of the committed
/// segment, each paired with its sequence's transpose action at `bit_point`.
pub(super) fn bit_pairs<W, F>(
    layout: &CompiledKeyLayout<W>,
    rows: &[F],
    bit_point: &[F],
) -> SumOfProducts<F>
where
    W: Word,
    F: Field,
{
    let sequences = layout.witness().shift_sequences();
    let witness_rows = rows
        .chunks(W::BITS as usize)
        .map(<[F]>::to_vec)
        .collect::<Vec<_>>();

    // The other factor is the transpose action of each fixed shift sequence.
    let output = Point::new(bit_point).equality_weights_msb();
    let pairs = witness_rows
        .into_iter()
        .zip(sequences)
        .map(|(witness, &shifts)| {
            [
                Poly::new(witness),
                Poly::new(shift_weights(shifts, &output)),
            ]
        })
        .collect();
    SumOfProducts::new(pairs, W::BITS.ilog2() as usize)
}

/// Builds the word-index sumcheck after the bit axis is bound.
pub(super) fn word_prover<W, F>(
    layout: &CompiledKeyLayout<W>,
    values: &PackedWitness<W>,
    prepared: &PreparedWeights<F>,
    bit_point: &[F],
    shift_evaluations: &[F],
) -> SumOfProducts<F>
where
    W: PackedWord,
    F: Field + Algebra<Gf2>,
{
    // The bit point folds each packed source word directly to one challenge-field element.
    let tables = byte_tables(&Point::new(bit_point).equality_weights_msb());
    let mut witness = values
        .witness()
        .par_iter()
        .map(|&packed| evaluate_bytes(W::unpack(packed).to_u64(), &tables))
        .collect::<Vec<_>>();

    // The public wiring contributes one scalar per committed word.
    let mut wiring = (0..values.witness().len())
        .into_par_iter()
        .map(|word| word_weight(layout.witness(), word, prepared, shift_evaluations))
        .collect::<Vec<_>>();

    // Both tables share the smallest power-of-two cube covering the committed segment.
    let capacity = witness.len().max(1).next_power_of_two();
    witness.resize(capacity, F::ZERO);
    wiring.resize(capacity, F::ZERO);
    SumOfProducts::new(
        vec![[Poly::new(witness), Poly::new(wiring)]],
        capacity.ilog2() as usize,
    )
}

/// Evaluates the committed wiring polynomial at a word point.
///
/// An interleaved block costs its gadget's references, whatever its instance count: the
/// equality weights of its words and relations factor into the instance's low bits, which
/// sum to one product over the block, and the slot's and relation's high bits.
pub(super) fn wiring_evaluation<W, F>(
    layout: &CompiledKeyLayout<W>,
    prepared: &PreparedWeights<F>,
    shift_evaluations: &[F],
    word_point: &[F],
) -> F
where
    W: Word,
    F: Field + Algebra<Gf2>,
{
    let segment = layout.witness();
    let blocks = segment.block_views().collect::<Vec<_>>();
    let flat = blocks
        .iter()
        .any(|block| block.interleaved.is_none_or(|log| log == 0));
    // Missing words in the final power-of-two domain carry zero wiring weight.
    let word_weights = flat.then(|| Point::new(word_point).equality_weights_msb());
    blocks
        .into_par_iter()
        .map(|block| match block.interleaved {
            Some(log) if log > 0 => {
                interleaved_weight(segment, &block, log, prepared, shift_evaluations, word_point)
            }
            _ => {
                let word_weights = word_weights
                    .as_ref()
                    .expect("the word table is built when a block reads it");
                (block.word_base..block.word_base + block.slots * block.instances)
                    .into_par_iter()
                    .map(|word| {
                        word_weights[word]
                            * word_weight(segment, word, prepared, shift_evaluations)
                    })
                    .sum()
            }
        })
        .sum()
}

/// The equality weight at `point` of the index `index`, its leading bit matched to the
/// point's first coordinate.
fn equality_at<F: Field>(point: &[F], index: usize) -> F {
    let n = point.len();
    point
        .iter()
        .enumerate()
        .map(|(i, &r)| {
            if (index >> (n - 1 - i)) & 1 == 1 {
                r
            } else {
                F::ONE - r
            }
        })
        .product()
}

/// One interleaved block of `2^log` instances' share of the wiring evaluation.
///
/// Its word `base + slot * 2^log + i` and relation `base_f + relation * 2^log + i` share the
/// low `log` bits `i`, both bases being multiples of `2^log`, so the sum over `i` is the
/// product of the low coordinates' equality factors.
fn interleaved_weight<W: Word, F: Field>(
    segment: &CompiledSegment<W>,
    block: &BlockView,
    log: usize,
    prepared: &PreparedWeights<F>,
    shift_evaluations: &[F],
    word_point: &[F],
) -> F {
    let constraint_point = &prepared.point;
    if log > word_point.len() || log > constraint_point.len() {
        return F::ZERO;
    }
    let (word_high, word_low) = word_point.split_at(word_point.len() - log);
    let (row_high, row_low) = constraint_point.split_at(constraint_point.len() - log);
    let low = word_low
        .iter()
        .zip(row_low)
        .map(|(&w, &r)| w * r + (F::ONE - w) * (F::ONE - r))
        .product::<F>();

    let rows: [Vec<F>; 3] = core::array::from_fn(|family| {
        let first = block.constraint_base[family] >> log;
        (0..block.strides[family])
            .map(|relation| equality_at(row_high, first + relation))
            .collect()
    });
    let first_word = block.word_base >> log;
    let high = (0..block.slots)
        .into_par_iter()
        .map(|slot| {
            let keys = segment
                .slot_keys(block.slot_base + slot)
                .map(|key| {
                    let family = key.operation().code() as usize;
                    let occurrences = key
                        .references()
                        .map(|reference| {
                            let operand = operand_index(key.operation(), reference.operand());
                            prepared.operand[operand] * rows[family][reference.constraint() as usize]
                        })
                        .sum::<F>();
                    prepared.operation[family]
                        * occurrences
                        * shift_evaluations[key.shift_index() as usize]
                })
                .sum::<F>();
            equality_at(word_high, first_word + slot) * keys
        })
        .sum::<F>();
    low * high
}

/// Accumulates every compiled key attached to one source word.
fn word_weight<W: Word, F: Field>(
    segment: &CompiledSegment<W>,
    word: usize,
    prepared: &PreparedWeights<F>,
    shift_evaluations: &[F],
) -> F {
    // A key coefficient factorizes into wiring, batching, and shift evaluations.
    segment
        .keys(word)
        .expect("the checked segment contains this word")
        .map(|key| prepared.key(&key) * shift_evaluations[key.shift_index() as usize])
        .sum()
}

/// Evaluates every fixed shift operator at a sampled input-bit point.
pub(super) fn shift_evaluations<W, F>(
    layout: &CompiledKeyLayout<W>,
    output_point: &[F],
    input_point: &[F],
) -> Vec<F>
where
    W: Word,
    F: Field,
{
    // Each small operator row is evaluated independently at the same input point.
    let output = Point::new(output_point).equality_weights_msb();
    let input = Point::new(input_point).equality_weights_msb();
    layout
        .witness()
        .shift_sequences()
        .iter()
        .map(|&shifts| F::batched_linear_combination(&shift_weights(shifts, &output), &input))
        .collect()
}

/// Computes claims directly from a checked witness for reduction tests.
#[cfg(test)]
pub(super) fn evaluate_claim<W, F>(
    system: &ConstraintSystem<W>,
    values: &PackedWitness<W>,
    constraint_point: Vec<F>,
    bit_point: Vec<F>,
) -> ShiftClaim<F>
where
    W: PackedWord,
    F: Field + Algebra<Gf2>,
{
    // Constraint rows are zero-padded to the widest relation family.
    let constraint_weights = Point::new(constraint_point.as_slice()).equality_weights_msb();
    let bit_weights = Point::new(bit_point.as_slice()).equality_weights_msb();
    let public = values
        .public()
        .iter()
        .copied()
        .map(W::unpack)
        .collect::<Vec<_>>();
    let witness = values
        .witness()
        .iter()
        .copied()
        .map(W::unpack)
        .collect::<Vec<_>>();
    let evaluate = |operand: &p3_word::Operand<W>| {
        let word = operand
            .evaluate(&public, &witness)
            .expect("a checked system contains only in-bounds terms");
        evaluate_word(word, &bit_weights)
    };
    // Each semantic column is evaluated independently at the shared points.
    let zero = [system
        .zero_constraints()
        .iter()
        .enumerate()
        .map(|(row, relation)| constraint_weights[row] * evaluate(relation.value()))
        .sum()];
    let mut bitwise_and = [F::ZERO; 3];
    for (row, relation) in system.and_constraints().iter().enumerate() {
        let weight = constraint_weights[row];
        bitwise_and[0] += weight * evaluate(relation.left());
        bitwise_and[1] += weight * evaluate(relation.right());
        bitwise_and[2] += weight * evaluate(relation.output());
    }

    let mut integer_mul = [F::ZERO; 4];
    for (row, relation) in system.integer_mul_constraints().iter().enumerate() {
        let weight = constraint_weights[row];
        integer_mul[0] += weight * evaluate(relation.left());
        integer_mul[1] += weight * evaluate(relation.right());
        integer_mul[2] += weight * evaluate(relation.low());
        integer_mul[3] += weight * evaluate(relation.high());
    }
    ShiftClaim::new(constraint_point, bit_point, zero, bitwise_and, integer_mul)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_field::{BinaryField128, TowerLevel};
    use p3_word::{ShiftKind, Word32, Word64};

    use super::*;

    type F = BinaryField128;

    /// Enumerates every distinct shift accepted by a 64-bit word.
    fn shifts() -> Vec<Shift<Word64>> {
        // Distance zero normalizes to one identity, so the other seven spellings are dropped.
        (0..8)
            .flat_map(|code| {
                let kind = ShiftKind::from_code(code).expect("codes below eight are assigned");
                let width = if kind.is_lane32() { 32 } else { 64 };
                let first = usize::from(code != 0);
                (first..width).map(move |amount| Shift::new(kind, amount).unwrap())
            })
            .collect()
    }

    /// Computes the transpose table by applying both slots directly to every basis bit.
    fn direct_weights(shifts: [Shift<Word64>; 2], output: &[F]) -> Vec<F> {
        (0..64)
            .map(|input| {
                let basis = Word64::from_low_bits(1_u64 << input);
                let shifted = shifts[1].apply(shifts[0].apply(basis)).to_u64();
                (0..64)
                    .filter(|&bit| (shifted >> bit) & 1 == 1)
                    .map(|bit| output[bit])
                    .sum()
            })
            .collect()
    }

    /// Compares both transpose paths over the supplied slot pairs.
    fn check_pairs(pairs: impl Iterator<Item = [Shift<Word64>; 2]>) {
        // Distinct coefficients expose bit order, lane boundaries, and sign-extension fan-out.
        let output = (0..64)
            .map(|bit| F::from_repr(1_u128 << bit))
            .collect::<Vec<_>>();
        for sequence in pairs {
            assert_eq!(
                shift_weights(sequence, &output),
                direct_weights(sequence, &output)
            );
        }
    }

    #[test]
    fn word32_transpose_matches_every_single_shift_and_edge_composition() {
        let output = (0..32)
            .map(|bit| F::from_repr(1_u128 << bit))
            .collect::<Vec<_>>();
        for code in 0..4 {
            let kind = ShiftKind::from_code(code).unwrap();
            for amount in 0..32 {
                let inner = Shift::<Word32>::new(kind, amount).unwrap();
                for outer_kind in 0..4 {
                    for outer_amount in [0, 1, 15, 31] {
                        let outer =
                            Shift::new(ShiftKind::from_code(outer_kind).unwrap(), outer_amount)
                                .unwrap();
                        let expected = (0..32)
                            .map(|input| {
                                let basis = Word32::from_low_bits(1_u64 << input);
                                let shifted = outer.apply(inner.apply(basis)).to_u64();
                                (0..32)
                                    .filter(|&bit| (shifted >> bit) & 1 == 1)
                                    .map(|bit| output[bit])
                                    .sum::<F>()
                            })
                            .collect::<Vec<_>>();
                        assert_eq!(shift_weights([inner, outer], &output), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn factorized_transpose_matches_a_sampled_shift_pair_grid() {
        // Strides coprime to the count visit every shift in both slots, three pairings each.
        let shifts = shifts();
        let count = shifts.len();
        let mut pairs = Vec::with_capacity(3 * count);
        for stride in [1_usize, 31, 137] {
            for inner in 0..count {
                pairs.push([shifts[inner], shifts[(inner * stride + 1) % count]]);
            }
        }
        check_pairs(pairs.into_iter());
    }

    #[test]
    #[ignore = "the full square is 142k pairs and dominates the default test profile"]
    fn factorized_transpose_matches_every_shift_pair() {
        // Exhausting both slots catches reversed composition and every legal kind and amount.
        let shifts = shifts();
        check_pairs(
            shifts
                .iter()
                .flat_map(|&inner| shifts.iter().map(move |&outer| [inner, outer])),
        );
    }

    /// A gadget of every relation family with shifted operands, `width` private words wide.
    fn interleaving_gadget(width: usize) -> p3_word::Component<Word64> {
        use p3_word::{
            AndConstraint, IntegerMulConstraint, Operand, ShiftedValue, ValueIndex,
            ZeroConstraint,
        };
        let input = ValueIndex::public(0).unwrap();
        let local = |slot: usize| ValueIndex::witness(slot).unwrap();
        let rotated = |slot: usize, amount: usize| {
            ShiftedValue::single(local(slot), Shift::new(ShiftKind::RotateRight, amount).unwrap())
        };
        let zero = (1..width)
            .map(|slot| {
                ZeroConstraint::new(Operand::new(alloc::vec![
                    ShiftedValue::plain(local(slot)),
                    rotated(slot - 1, slot),
                ]))
            })
            .collect();
        let and = (0..width)
            .map(|slot| {
                AndConstraint::new(
                    Operand::single(ShiftedValue::plain(input)),
                    Operand::single(rotated(slot, 3)),
                    Operand::single(ShiftedValue::plain(local((slot + 1) % width))),
                )
            })
            .collect();
        let product = alloc::vec![IntegerMulConstraint::new(
            Operand::single(ShiftedValue::plain(local(0))),
            Operand::single(rotated(width - 1, 7)),
            Operand::single(ShiftedValue::plain(local(1))),
            Operand::single(ShiftedValue::plain(input)),
        )];
        let body = ConstraintSystem::new(1, width, zero, and, product).unwrap();
        p3_word::Component::new(body, 1, 0).unwrap()
    }

    #[test]
    fn an_interleaved_wiring_evaluates_as_its_lowered_statement() {
        use p3_word::{ComponentCall, Composition};

        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut sample = || {
            let mut word = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            F::from_repr(u128::from(word()) << 64 | u128::from(word()))
        };
        for counts in [[5_usize, 3], [1, 1], [8, 6], [0, 7]] {
            let composition = Composition::interleaved(alloc::vec![
                ComponentCall::new(interleaving_gadget(3), counts[0]),
                ComponentCall::new(interleaving_gadget(5), counts[1]),
            ])
            .unwrap();
            let composed = CompiledKeyLayout::compose(&composition).unwrap();
            let flat = CompiledKeyLayout::new(&composition.lower().unwrap()).unwrap();
            assert_eq!(composed.witness().len(), flat.witness().len());

            let rows = composition.relation_counts().into_iter().max().unwrap();
            let constraint_point = (0..rows.next_power_of_two().ilog2())
                .map(|_| sample())
                .collect::<Vec<_>>();
            let words = composition.witness_len().max(1).next_power_of_two().ilog2();
            let word_point = (0..words).map(|_| sample()).collect::<Vec<_>>();
            let output = (0..6).map(|_| sample()).collect::<Vec<_>>();
            let input = (0..6).map(|_| sample()).collect::<Vec<_>>();
            let operation = (0..3).map(|_| sample()).collect::<Vec<_>>();
            let operand = (0..8).map(|_| sample()).collect::<Vec<_>>();

            let evaluate = |layout: &CompiledKeyLayout<Word64>| {
                let prepared = PreparedWeights::from_parts(
                    &constraint_point,
                    operation.clone(),
                    operand.clone(),
                );
                let shifts = shift_evaluations(layout, &output, &input);
                wiring_evaluation(layout, &prepared, &shifts, &word_point)
            };
            assert_eq!(evaluate(&composed), evaluate(&flat), "counts {counts:?}");
        }
    }
}
