//! Packed relation columns derived from a checked witness.

use alloc::vec::Vec;
use core::array;

#[cfg(test)]
use p3_field::Field;
use p3_maybe_rayon::prelude::*;
use p3_word::{CallBlock, Composition, ConstraintSystem, Operand, Placement, ValueIndex};

use crate::statement::Statement;
use crate::{Packed, PackedWitness, PackedWord, WitnessError};

/// Packed rows used by the relation reductions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationColumns<W: PackedWord> {
    /// The operand each linear relation requires to vanish.
    zero: Vec<Packed<W>>,
    /// The left input, right input, and output of each bitwise product.
    bitwise_and: [Vec<Packed<W>>; 3],
    /// The factors and result limbs of each unsigned product.
    integer_mul: [Vec<Packed<W>>; 4],
}

impl<W: PackedWord> OperationColumns<W> {
    /// Evaluates the operands needed by the nonlinear reductions.
    ///
    /// A composed statement walks each gadget once per instance.
    ///
    /// A relation's column position is therefore exactly its composed relation index.
    ///
    /// # Errors
    ///
    /// Returns an error when the witness does not have the statement's shape.
    pub fn new(statement: &Statement<W>, values: &PackedWitness<W>) -> Result<Self, WitnessError> {
        values.check_shape(statement)?;
        let counts = statement.relation_counts();
        let mut columns = Self {
            zero: Vec::with_capacity(counts[0]),
            bitwise_and: array::from_fn(|_| Vec::with_capacity(counts[1])),
            integer_mul: array::from_fn(|_| Vec::with_capacity(counts[2])),
        };

        match statement {
            // A flat statement addresses the composed segments directly.
            Statement::Flat(system) => columns.extend(system, values, |index| index),
            Statement::Composed(composition)
                if composition.placement() == Placement::Interleaved =>
            {
                for block in composition.blocks() {
                    columns.extend_block(composition, block, values);
                }
            }
            Statement::Composed(composition) => {
                for (call, index) in composition.calls().iter().zip(0_usize..) {
                    let component = call.component();
                    for instance in 0..call.instances() {
                        // The affine map is the same one the compiled layout reverses.
                        columns.extend(component.body(), values, |slot| {
                            composition
                                .resolve(index, instance, slot)
                                .expect("a checked body only names slots the call declares")
                        });
                    }
                }
            }
        }
        Ok(columns)
    }

    /// Appends one interleaved block: each family's relations by the gadget's relation, then
    /// by the instance.
    fn extend_block(
        &mut self,
        composition: &Composition<W>,
        block: &CallBlock,
        values: &PackedWitness<W>,
    ) {
        let body = composition.calls()[block.call()].component().body();
        let instances = block.instances();
        let evaluate = |operand: &Operand<W>, local: usize| {
            let mut result = W::pack(W::ZERO);
            for term in operand.terms() {
                let at = composition
                    .resolve(block.call(), block.first_instance() + local, term.index())
                    .expect("a checked body only names slots the call declares");
                let word = values
                    .get(at)
                    .expect("the checked system only contains in-bounds indices");
                result += W::pack(term.apply(word));
            }
            result
        };
        let rows = |operands: Vec<&Operand<W>>| {
            (0..operands.len() * instances)
                .into_par_iter()
                .map(|row| evaluate(operands[row / instances], row % instances))
                .collect::<Vec<_>>()
        };

        let zero = body.zero_constraints();
        self.zero.extend(rows(zero.iter().map(|c| c.value()).collect()));

        let and = body.and_constraints();
        let [left, right, output] = &mut self.bitwise_and;
        left.extend(rows(and.iter().map(|c| c.left()).collect()));
        right.extend(rows(and.iter().map(|c| c.right()).collect()));
        output.extend(rows(and.iter().map(|c| c.output()).collect()));

        let mul = body.integer_mul_constraints();
        let [left, right, low, high] = &mut self.integer_mul;
        left.extend(rows(mul.iter().map(|c| c.left()).collect()));
        right.extend(rows(mul.iter().map(|c| c.right()).collect()));
        low.extend(rows(mul.iter().map(|c| c.low()).collect()));
        high.extend(rows(mul.iter().map(|c| c.high()).collect()));
    }

    /// Appends one relation block, readdressing every term as it is read.
    fn extend(
        &mut self,
        system: &ConstraintSystem<W>,
        values: &PackedWitness<W>,
        remap: impl Fn(ValueIndex) -> ValueIndex + Sync,
    ) {
        let evaluate = |operand: &Operand<W>| {
            let mut result = W::pack(W::ZERO);
            for term in operand.terms() {
                let word = values
                    .get(remap(term.index()))
                    .expect("the checked system only contains in-bounds indices");
                result += W::pack(term.apply(word));
            }
            result
        };

        // Each column is one parallel pass over its family, appended in relation order.
        let fill = |column: &mut Vec<Packed<W>>, operands: Vec<Packed<W>>| {
            if column.is_empty() {
                *column = operands;
            } else {
                column.extend(operands);
            }
        };

        // The linear family has one semantic operand per relation.
        let zero = system.zero_constraints();
        fill(
            &mut self.zero,
            zero.par_iter()
                .map(|constraint| evaluate(constraint.value()))
                .collect(),
        );

        // The bitwise reduction reads the left, right, and output operands.
        let and = system.and_constraints();
        let [left, right, output] = &mut self.bitwise_and;
        fill(left, and.par_iter().map(|c| evaluate(c.left())).collect());
        fill(right, and.par_iter().map(|c| evaluate(c.right())).collect());
        fill(
            output,
            and.par_iter().map(|c| evaluate(c.output())).collect(),
        );

        // The integer products read both factors, then the low and high limbs.
        let mul = system.integer_mul_constraints();
        let [left, right, low, high] = &mut self.integer_mul;
        fill(left, mul.par_iter().map(|c| evaluate(c.left())).collect());
        fill(right, mul.par_iter().map(|c| evaluate(c.right())).collect());
        fill(low, mul.par_iter().map(|c| evaluate(c.low())).collect());
        fill(high, mul.par_iter().map(|c| evaluate(c.high())).collect());
    }

    /// Returns the operand of each linear relation.
    #[inline]
    pub fn zero(&self) -> &[Packed<W>] {
        &self.zero
    }

    /// Returns the left, right, and output AND operands.
    #[inline]
    pub const fn bitwise_and(&self) -> &[Vec<Packed<W>>; 3] {
        &self.bitwise_and
    }

    /// Returns the factors followed by the low and high product limbs.
    #[inline]
    pub const fn integer_mul(&self) -> &[Vec<Packed<W>>; 4] {
        &self.integer_mul
    }
}

/// Expands one packed column into its bit multilinear over the padded cube.
#[cfg(test)]
pub(crate) fn bit_table<W: PackedWord, EF: Field>(column: &[Packed<W>], rows: usize) -> Vec<EF> {
    // Row index leads the flat address and within-word bit index trails it.
    let width = W::BITS as usize;
    let mut table = EF::zero_vec(rows * width);
    for (row, &packed) in column.iter().enumerate() {
        let mut remaining = W::unpack(packed).to_u64();
        while remaining != 0 {
            // Visit only set lanes, which halves the scatter on random words.
            let bit = remaining.trailing_zeros() as usize;
            table[row * width + bit] = EF::ONE;
            remaining &= remaining - 1;
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;
    use p3_word::{
        AndConstraint, ConstraintSystem, IntegerMulConstraint, Operand, Shift, ShiftKind,
        ShiftedValue, ValueIndex, Word32, Word64,
    };
    use proptest::prelude::*;

    use super::*;

    fn operand(terms: Vec<ShiftedValue<Word64>>) -> Operand<Word64> {
        Operand::new(terms)
    }

    #[test]
    fn packed_columns_match_scalar_operand_evaluation() {
        let public_index = ValueIndex::public(0).unwrap();
        let witness_index = ValueIndex::witness(0).unwrap();
        let shift = Shift::new(ShiftKind::Lane32RotateRight, 7).unwrap();
        let left = operand(vec![ShiftedValue::single(witness_index, shift)]);
        let right = operand(vec![ShiftedValue::plain(public_index)]);
        let output = operand(vec![]);
        let and = AndConstraint::new(left.clone(), right.clone(), output);
        let mul =
            IntegerMulConstraint::new(left.clone(), right.clone(), left.clone(), right.clone());
        let system = ConstraintSystem::new(1, 1, vec![], vec![and], vec![mul]).unwrap();
        let public = [Word64::new(0x0102_0304_0506_0708)];
        let witness = [Word64::new(0xfedc_ba98_7654_3210)];
        let packed = PackedWitness::new(&system, &public, &witness).unwrap();
        let columns = OperationColumns::new(&Statement::from(system), &packed).unwrap();

        assert_eq!(
            columns.bitwise_and()[0][0].to_bits(),
            left.evaluate(&public, &witness).unwrap().get()
        );
        assert_eq!(
            columns.bitwise_and()[1][0].to_bits(),
            right.evaluate(&public, &witness).unwrap().get()
        );
        for (column, expected) in columns
            .integer_mul()
            .iter()
            .zip([&left, &right, &left, &right])
        {
            assert_eq!(
                column[0].to_bits(),
                expected.evaluate(&public, &witness).unwrap().get()
            );
        }
    }

    #[test]
    fn empty_families_have_no_protocol_padding() {
        let system = ConstraintSystem::<Word64>::new(0, 0, vec![], vec![], vec![]).unwrap();
        let packed = PackedWitness::new(&system, &[], &[]).unwrap();
        let columns = OperationColumns::new(&Statement::from(system), &packed).unwrap();

        assert!(columns.bitwise_and().iter().all(Vec::is_empty));
        assert!(columns.integer_mul().iter().all(Vec::is_empty));
    }

    #[test]
    fn rejects_a_witness_packed_for_another_shape() {
        let source = ConstraintSystem::<Word64>::new(0, 0, vec![], vec![], vec![]).unwrap();
        let target = ConstraintSystem::<Word64>::new(0, 1, vec![], vec![], vec![]).unwrap();
        let packed = PackedWitness::new(&source, &[], &[]).unwrap();

        assert_eq!(
            OperationColumns::new(&Statement::from(target), &packed),
            Err(WitnessError {
                segment: p3_word::Segment::Witness,
                expected: 1,
                actual: 0,
            })
        );
    }

    proptest! {
        #[test]
        fn random_packed_columns_match_the_scalar_reference(
            public_value in any::<u64>(),
            witness_value in any::<u64>(),
            full_amount in 1usize..64,
            lane_amount in 1usize..32,
        ) {
            let public_index = ValueIndex::public(0).unwrap();
            let witness_index = ValueIndex::witness(0).unwrap();
            let full = Shift::new(ShiftKind::RotateRight, full_amount).unwrap();
            let lanes = Shift::new(ShiftKind::Lane32ArithmeticRight, lane_amount).unwrap();
            let left = operand(vec![
                ShiftedValue::single(witness_index, full),
                ShiftedValue::plain(public_index),
            ]);
            let right = operand(vec![ShiftedValue::single(public_index, lanes)]);
            let output = operand(vec![ShiftedValue::plain(witness_index)]);
            let and = AndConstraint::new(left.clone(), right.clone(), output.clone());
            let mul = IntegerMulConstraint::new(
                left.clone(),
                right.clone(),
                output.clone(),
                left.clone(),
            );
            let system = ConstraintSystem::new(1, 1, vec![], vec![and], vec![mul]).unwrap();
            let public = [Word64::new(public_value)];
            let witness = [Word64::new(witness_value)];
            let packed = PackedWitness::new(&system, &public, &witness).unwrap();
            let columns = OperationColumns::new(&Statement::from(system), &packed).unwrap();

            for (column, expected) in columns
                .bitwise_and()
                .iter()
                .zip([&left, &right])
            {
                prop_assert_eq!(
                    column[0].to_bits(),
                    expected.evaluate(&public, &witness).unwrap().get()
                );
            }
            for (column, expected) in columns
                .integer_mul()
                .iter()
                .zip([&left, &right, &output, &left])
            {
                prop_assert_eq!(
                    column[0].to_bits(),
                    expected.evaluate(&public, &witness).unwrap().get()
                );
            }
        }
    }

    #[test]
    fn a_packed_column_expands_to_its_set_bits() {
        // Fixture state: one 32-bit word with its lowest and highest bits set.
        let column = [Word32::pack(Word32::new(0x8000_0001))];
        let table = bit_table::<Word32, BinaryField128>(&column, 2);

        // Set lanes are one, every other lane of both rows is zero.
        assert_eq!(table.len(), 64);
        for (index, value) in table.iter().enumerate() {
            let expected = if index == 0 || index == 31 {
                BinaryField128::ONE
            } else {
                BinaryField128::ZERO
            };
            assert_eq!(*value, expected, "lane {index}");
        }
    }

    #[test]
    fn a_padded_row_contributes_no_bits() {
        // Fixture state: one 64-bit row inside a two-row padded cube.
        let column = [Word64::pack(Word64::new(u64::MAX))];
        let table = bit_table::<Word64, BinaryField128>(&column, 2);

        assert!(
            table[..64]
                .iter()
                .all(|value| *value == BinaryField128::ONE)
        );
        assert!(
            table[64..]
                .iter()
                .all(|value| *value == BinaryField128::ZERO)
        );
    }
}
