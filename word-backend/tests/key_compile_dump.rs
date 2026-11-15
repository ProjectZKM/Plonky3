//! Key compilation timed on a word statement the prover dumped, its compiled key fingerprinted
//! so a faster compiler can be checked to make the same key.

use std::hash::Hasher;
use std::time::Instant;

use p3_word::{
    AndConstraint, ConstraintSystem, IntegerMulConstraint, Operand, Shift, ShiftKind,
    ShiftedValue, ValueIndex, Word64, ZeroConstraint,
};
use p3_word_backend::WordProofKey;

struct Reader<'a> {
    bytes: &'a [u8],
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let (head, tail) = self.bytes.split_at(N);
        self.bytes = tail;
        head.try_into().unwrap()
    }

    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take())
    }

    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take())
    }

    fn shift(kind: u8, amount: u8) -> Shift<Word64> {
        Shift::new(ShiftKind::from_code(kind).unwrap(), amount as usize).unwrap()
    }

    fn operand(&mut self) -> Operand<Word64> {
        let count = self.u32() as usize;
        let terms = (0..count)
            .map(|_| {
                let at = self.u32();
                let index = if at >> 31 == 1 {
                    ValueIndex::witness((at & !(1 << 31)) as usize)
                } else {
                    ValueIndex::public(at as usize)
                }
                .unwrap();
                let [ik, ia, ok, oa] = self.take::<4>();
                match (ia, oa) {
                    (0, 0) => ShiftedValue::plain(index),
                    (_, 0) => ShiftedValue::single(index, Self::shift(ik, ia)),
                    _ => ShiftedValue::pair(index, Self::shift(ik, ia), Self::shift(ok, oa)).unwrap(),
                }
            })
            .collect();
        Operand::new(terms)
    }
}

/// The statement `ZIREN_WORD_DUMP` wrote: counts, words, then every operand in family order.
fn read(bytes: &[u8]) -> ConstraintSystem<Word64> {
    let mut reader = Reader { bytes };
    let [public, witness, zero, and, mul] = core::array::from_fn(|_| reader.u64() as usize);
    reader.bytes = &reader.bytes[8 * (public + witness)..];
    let zero = (0..zero).map(|_| ZeroConstraint::new(reader.operand())).collect();
    let and = (0..and)
        .map(|_| AndConstraint::new(reader.operand(), reader.operand(), reader.operand()))
        .collect();
    let mul = (0..mul)
        .map(|_| {
            IntegerMulConstraint::new(
                reader.operand(),
                reader.operand(),
                reader.operand(),
                reader.operand(),
            )
        })
        .collect();
    assert!(reader.bytes.is_empty(), "the whole dump read");
    ConstraintSystem::new(public, witness, zero, and, mul).unwrap()
}

#[test]
#[ignore = "reads the statement WORD_DUMP names"]
fn a_dumped_statement_compiles() {
    let _ = tracing_subscriber::fmt()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .try_init();
    let path = std::env::var("WORD_DUMP").expect("WORD_DUMP names a dumped statement");
    let started = Instant::now();
    let system = read(&std::fs::read(path).unwrap());
    let read_s = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let key = WordProofKey::<Word64>::new(system).unwrap();
    let compile_s = started.elapsed().as_secs_f64();
    let bytes = postcard::to_allocvec(&key).unwrap();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hasher.write(&bytes);
    println!(
        "read {read_s:.1} s, compile {compile_s:.2} s, key {} bytes, fingerprint {:016x}",
        bytes.len(),
        hasher.finish()
    );
}
