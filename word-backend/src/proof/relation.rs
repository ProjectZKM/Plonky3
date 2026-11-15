//! Batched vanishing check over the shared constraint and within-word cube.
//!
//! Both local families hold on one constraint row and one bit.
//!
//! ```text
//!     linear    value(row, bit) = 0
//!     bitwise   left(row, bit) * right(row, bit) = output(row, bit)
//! ```
//!
//! A statement with products adds five more terms under further powers of the same coefficient.
//!
//! ```text
//!     low bit   a(row, 0) * b(row, 0) = lo(row, 0)
//!     claims    a, b, lo, hi at the points the multiplication reduction ended on
//! ```
//!
//! The low bit closes the one collision the exponent lift leaves open.
//!
//! The four claims share this sumcheck, so every operand claim ends at one point.
//!
//! Padding rows hold the empty operand, which satisfies every shape.
//!
//! That shared point is the one the shift reduction consumes.
//!
//! # Variable order
//!
//! The within-word variables are bound first, the constraint rows after them.
//!
//! Every weight is an equality factor, so it splits into a row table and a bit table.
//!
//! ```text
//!     eq(point, (row, bit)) = eq(point_row, row) * eq(point_bit, bit)
//! ```
//!
//! While bits are bound, each column stays a packed word per row.
//!
//! - A word folds through byte tables of the bound prefix's equality weights.
//! - Linear terms are summed over the rows once, into one table per bit.
//! - Only the two products are recomputed per row and per round.
//!
//! Once every bit is bound, each column is one field element per row.
//!
//! The memory is therefore a few field elements per row rather than one per bit.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::array;

use p3_field::Field;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_sumcheck::generic_degree::RoundProver;
use tracing::info_span;

use crate::device::{
    DeviceRelation, PRODUCT_COLUMNS, WordDevice, Words, from_device, slice_to_device, to_device,
};

/// Per-variable degree of the batched relation polynomial.
///
/// One equality factor multiplies a product of two operand columns.
pub(super) const ZEROCHECK_DEGREE: usize = 3;

/// Terms batched when the statement declares no product: linear, then bitwise.
pub(super) const BASE_TERMS: usize = 2;

/// Terms batched when it does: the two above, the low bit, and four operand claims.
pub(super) const PRODUCT_TERMS: usize = 7;

/// Operand evaluations the vanishing check leaves for the shift reduction.
///
/// The vanishing operand comes first, then the bitwise left, right, and output operands.
///
/// The left factor, right factor, low limb, and high limb follow.
pub(super) const OPERAND_EVALUATIONS: usize = 8;

/// Equality weights of the four product terms, in low-bit, left, right, then limb order.
pub(super) type ProductWeights<F> = [F; 4];

/// Rows one parallel task covers.
const BLOCK: usize = 1 << 12;

/// Linear terms summed over the rows into one table per bit.
///
/// The order is the vanishing operand, the bitwise output, and the low limb under the
/// vanishing weight, then the left factor, the right factor, and both limbs under their claims.
const SUMS: usize = 7;

/// The word columns of the check, one word per declared row.
///
/// Rows past a column's end are zero, as padding rows are.
pub(super) struct RelationWords {
    /// The operand every linear relation requires to vanish.
    pub(super) linear: Vec<u64>,
    /// The left input, right input, and output of every bitwise product.
    pub(super) bitwise: [Vec<u64>; 3],
    /// The left factor, right factor, low limb, and high limb of every product.
    pub(super) product: Option<[Vec<u64>; 4]>,
}

/// Rows past this word table's end are zero.
fn word(column: &[u64], row: usize) -> u64 {
    column.get(row).copied().unwrap_or(0)
}

/// Subset sums of up to eight weights, indexed by the byte selecting them.
fn byte_table<F: Field>(weights: &[F]) -> Vec<F> {
    debug_assert!(weights.len() <= 8);
    let mut table = F::zero_vec(1 << weights.len());
    for subset in 1..table.len() {
        // Each subset extends the one without its lowest member.
        table[subset] = table[subset & (subset - 1)] + weights[subset.trailing_zeros() as usize];
    }
    table
}

/// Folds words over a bound prefix of their bit variables.
///
/// With `stride` positions left, bit `prefix * stride + position` lies under prefix weight
/// `prefix`, so position `s` of the fold is the weighted sum of the bits `p * stride + s`.
struct WordFold<F> {
    /// Byte tables over consecutive groups of eight prefixes.
    tables: Vec<Vec<F>>,
    /// The number of positions each prefix spans.
    stride: usize,
}

impl<F: Field> WordFold<F> {
    /// Builds the fold for `prefix` weights over words of `width` bits.
    fn new(prefix: &[F], width: usize) -> Self {
        Self {
            tables: prefix.chunks(8).map(byte_table).collect(),
            stride: width / prefix.len(),
        }
    }

    /// Returns position `position` of the folded word.
    fn at(&self, word: u64, position: usize) -> F {
        let mut value = F::ZERO;
        for (group, table) in self.tables.iter().enumerate() {
            let mut byte = 0;
            for member in 0..table.len().trailing_zeros() as usize {
                let bit = (8 * group + member) * self.stride + position;
                byte |= (((word >> bit) & 1) as usize) << member;
            }
            value += table[byte];
        }
        value
    }

    /// Folds a table of one field element per bit the same way.
    fn table(&self, values: &[F], prefix: &[F]) -> Vec<F> {
        (0..self.stride)
            .map(|position| {
                prefix
                    .iter()
                    .enumerate()
                    .map(|(index, &weight)| weight * values[index * self.stride + position])
                    .sum()
            })
            .collect()
    }
}

/// Evaluates the line through a fold's two halves at `x`, at one remaining position.
fn line<F: Field>(folded: &[F], position: usize, x: F) -> F {
    let half = folded.len() / 2;
    folded[position] + x * (folded[half + position] - folded[position])
}

/// Adds two equal-length accumulators.
fn add_into<F: Field>(mut acc: Vec<F>, other: Vec<F>) -> Vec<F> {
    for (slot, value) in acc.iter_mut().zip(other) {
        *slot += value;
    }
    acc
}

/// Returns the three interpolation nodes the round polynomial is sent at.
fn nodes<F: Field>() -> [F; 3] {
    [F::ZERO, F::interpolation_node(2), F::interpolation_node(3)]
}

/// The product terms' weights, split into their row and bit tables.
struct ProductWeightTables<F> {
    /// Row tables of the left, right, and limb claim points.
    ///
    /// The low-bit term reads every row under the vanishing weight.
    rows: [Vec<F>; 3],
    /// Bit tables of the low-bit, left, right, and limb points.
    bits: [Vec<F>; 4],
}

/// The tables the row rounds bind, one field element per row.
struct RowTables<F> {
    /// The vanishing check's row weights, then the three claim row weights when present.
    weights: Vec<Vec<F>>,
    /// The eight operand columns at the bound bit point, the products absent without them.
    columns: Vec<Vec<F>>,
    /// The bit weights at the bound bit point, scaled by their batching powers.
    ///
    /// The order is the vanishing term, its product, the low bit, then the three claims and
    /// the high limb.
    scales: [F; 7],
}

/// Prover state for the batched relation vanishing check.
pub(super) struct RelationZerocheck<F> {
    /// The within-word variables, bound first.
    bit_variables: usize,
    /// The packed operand words.
    words: RelationWords,
    /// Powers of the coefficient separating every term, starting at one.
    powers: [F; PRODUCT_TERMS],
    /// Row weights of the vanishing point.
    row_equality: Vec<F>,
    /// Bit weights of the vanishing point.
    bit_equality: Vec<F>,
    /// The product terms' weights, when the statement declares products.
    product: Option<ProductWeightTables<F>>,
    /// The linear terms summed over the rows, one table per bit each.
    sums: [Vec<F>; SUMS],
    /// The bit challenges bound so far.
    bound: Vec<F>,
    /// The row-round state, once every bit is bound.
    rows: Option<RowTables<F>>,
    /// The device holding the columns and row weights, which then does every round.
    device: Option<Box<dyn DeviceRelation>>,
}

impl<F: Field> RelationZerocheck<F> {
    /// Creates a prover state over `row_variables` constraint rows and `bit_variables` bits.
    ///
    /// `points` are the four product points over rows then bits, when products are declared.
    /// Every round runs on `device` when there is one, which reads its own operation columns
    /// when `resident`; `words` then only say whether products are declared.
    #[allow(
        clippy::too_many_arguments,
        reason = "the check's inputs, each one"
    )]
    pub(super) fn new(
        vanishing_point: &[F],
        row_variables: usize,
        bit_variables: usize,
        words: RelationWords,
        points: Option<[Vec<F>; 4]>,
        batching: F,
        device: Option<&dyn WordDevice>,
        resident: bool,
    ) -> Self {
        debug_assert_eq!(vanishing_point.len(), row_variables + bit_variables);
        debug_assert_eq!(words.product.is_some(), points.is_some());
        let width = 1 << bit_variables;
        let device = device.map(|device| {
            fn column(resident: bool, index: usize, words: &[u64]) -> Words<'_> {
                if resident {
                    Words::Column(index)
                } else {
                    Words::Host(words)
                }
            }
            let mut columns = vec![column(resident, 0, &words.linear)];
            columns.extend(
                words
                    .bitwise
                    .iter()
                    .zip(1..)
                    .map(|(words, index)| column(resident, index, words)),
            );
            let mut row_points = vec![slice_to_device(&vanishing_point[..row_variables])];
            if let (Some(product), Some(points)) = (&words.product, &points) {
                columns.extend(
                    product
                        .iter()
                        .zip(PRODUCT_COLUMNS..)
                        .map(|(words, index)| column(resident, index, words)),
                );
                row_points.extend(
                    points[1..]
                        .iter()
                        .map(|p| slice_to_device(&p[..row_variables])),
                );
            }
            device.relation(&columns, &row_points, 1 << row_variables)
        });
        let on_host = device.is_none();
        let split = |point: &[F]| {
            let (rows, bits) = point.split_at(row_variables);
            (
                if on_host {
                    Point::new(rows).equality_weights_msb()
                } else {
                    Vec::new()
                },
                Point::new(bits).equality_weights_msb(),
            )
        };
        let (row_equality, bit_equality, product) = info_span!("weight tables").in_scope(|| {
            let (row_equality, bit_equality) = split(vanishing_point);
            let product = points.map(|points| {
                let [low, left, right, limb] = points.each_ref().map(|point| split(point));
                ProductWeightTables {
                    rows: [left.0, right.0, limb.0],
                    bits: [low.1, left.1, right.1, limb.1],
                }
            });
            (row_equality, bit_equality, product)
        });
        let sums = info_span!("linear sums").in_scope(|| match &device {
            Some(device) => {
                let sums = device.linear_sums(width);
                array::from_fn(|term| {
                    sums.get(term * width..(term + 1) * width).map_or_else(
                        || F::zero_vec(width),
                        |sum| sum.iter().map(|&v| from_device(v)).collect(),
                    )
                })
            }
            None => linear_sums(&words, &row_equality, product.as_ref(), width),
        });
        Self {
            bit_variables,
            words,
            powers: batching_powers(batching),
            row_equality,
            bit_equality,
            product,
            sums,
            bound: Vec::new(),
            rows: None,
            device,
        }
    }

    /// Returns the operand evaluations left once every variable is bound.
    pub(super) fn terminal_operands(&self) -> [F; OPERAND_EVALUATIONS] {
        let tables = self.rows.as_ref().expect("every bit variable was bound");
        let mut operands = [F::ZERO; OPERAND_EVALUATIONS];
        if let Some(device) = &self.device {
            for (slot, &value) in operands.iter_mut().zip(&device.terminal()) {
                *slot = from_device(value);
            }
            return operands;
        }
        for (slot, column) in operands.iter_mut().zip(&tables.columns) {
            debug_assert_eq!(column.len(), 1, "every row variable was bound");
            *slot = column[0];
        }
        operands
    }

    /// The round polynomial while a bit variable is free.
    fn bit_round(&self) -> Vec<F> {
        let _span = info_span!("bit round").entered();
        let width = 1 << self.bit_variables;
        let prefix = Point::new(self.bound.as_slice()).equality_weights_msb();
        let fold = WordFold::new(&prefix, width);
        let half = fold.stride / 2;
        let (products, low_bit) = match &self.device {
            Some(device) => {
                let values: Vec<F> = device
                    .bit_products(&slice_to_device(&prefix), width)
                    .into_iter()
                    .map(from_device)
                    .collect();
                let low_bit = self
                    .words
                    .product
                    .as_ref()
                    .map(|_| values[3 * half..].to_vec());
                (values[..3 * half].to_vec(), low_bit)
            }
            None => self.host_bit_products(&fold, half),
        };
        self.bit_round_poly(&fold, &prefix, half, &products, low_bit.as_deref())
    }

    /// The bitwise products of [`Self::bit_round`] and the low bit's, on the host.
    fn host_bit_products(&self, fold: &WordFold<F>, half: usize) -> (Vec<F>, Option<Vec<F>>) {
        let [left, right, _] = &self.words.bitwise;

        // Each remaining position's product, as the coefficients of a quadratic in x.
        let pairs = left.len().min(right.len());
        let products = (0..pairs.div_ceil(BLOCK)).into_par_iter().par_fold_reduce(
            || F::zero_vec(3 * half),
            |mut acc, block| {
                let start = block * BLOCK;
                for row in start..pairs.min(start + BLOCK) {
                    let (l, r) = (left[row], right[row]);
                    if l == 0 || r == 0 {
                        continue;
                    }
                    let weight = self.row_equality[row];
                    for position in 0..half {
                        accumulate_product(
                            &mut acc[3 * position..3 * position + 3],
                            weight,
                            [fold.at(l, position), fold.at(l, half + position)],
                            [fold.at(r, position), fold.at(r, half + position)],
                        );
                    }
                }
                acc
            },
            add_into,
        );

        // The low bit reads only the first position, where its bit weight lives.
        let low_bit = self.words.product.as_ref().map(|[a, b, _, _]| {
            let pairs = a.len().min(b.len());
            (0..pairs.div_ceil(BLOCK)).into_par_iter().par_fold_reduce(
                || F::zero_vec(3),
                |mut acc, block| {
                    let start = block * BLOCK;
                    for row in start..pairs.min(start + BLOCK) {
                        let (a, b) = (a[row], b[row]);
                        if a == 0 || b == 0 {
                            continue;
                        }
                        accumulate_product(
                            &mut acc,
                            self.row_equality[row],
                            [fold.at(a, 0), fold.at(a, half)],
                            [fold.at(b, 0), fold.at(b, half)],
                        );
                    }
                    acc
                },
                add_into,
            )
        });
        (products, low_bit)
    }

    /// The bit round polynomial from the folded products.
    fn bit_round_poly(
        &self,
        fold: &WordFold<F>,
        prefix: &[F],
        half: usize,
        products: &[F],
        low_bit: Option<&[F]>,
    ) -> Vec<F> {
        let folded = |values: &[F]| fold.table(values, prefix);
        let equality = folded(&self.bit_equality);
        let sums = self.sums.each_ref().map(|sum| folded(sum));
        let weights = self
            .product
            .as_ref()
            .map(|product| product.bits.each_ref().map(|bits| folded(bits)));
        let p = &self.powers;
        nodes::<F>()
            .into_iter()
            .map(|x| {
                let quadratic = |c: &[F]| c[0] + x * (c[1] + x * c[2]);
                (0..half)
                    .map(|position| {
                        let at = |table: &[F]| line(table, position, x);
                        let product = quadratic(&products[3 * position..3 * position + 3]);
                        let mut value =
                            at(&equality) * (at(&sums[0]) + p[1] * (product - at(&sums[1])));
                        if let (Some(weights), Some(low_bit)) = (&weights, &low_bit) {
                            let low = if position == 0 {
                                quadratic(low_bit)
                            } else {
                                F::ZERO
                            };
                            value += p[2] * at(&weights[0]) * (low - at(&sums[2]))
                                + p[3] * at(&weights[1]) * at(&sums[3])
                                + p[4] * at(&weights[2]) * at(&sums[4])
                                + at(&weights[3]) * (p[5] * at(&sums[5]) + p[6] * at(&sums[6]));
                        }
                        value
                    })
                    .sum()
            })
            .collect()
    }

    /// The round polynomial once only row variables are free.
    fn row_round(tables: &RowTables<F>) -> Vec<F> {
        let _span = info_span!("row round").entered();
        let half = tables.weights[0].len() / 2;
        let nodes = nodes::<F>();
        let sums = (0..half).into_par_iter().par_fold_reduce(
            || [F::ZERO; 3],
            |mut acc, row| {
                for (slot, &x) in acc.iter_mut().zip(&nodes) {
                    let at = |table: &Vec<F>| table[row] + x * (table[row + half] - table[row]);
                    // Absent product tables read as zero, which their zero scales ignore.
                    let mut weights = [F::ZERO; 4];
                    for (value, table) in weights.iter_mut().zip(&tables.weights) {
                        *value = at(table);
                    }
                    let mut columns = [F::ZERO; OPERAND_EVALUATIONS];
                    for (value, table) in columns.iter_mut().zip(&tables.columns) {
                        *value = at(table);
                    }
                    *slot += row_value(&weights, &columns, &tables.scales);
                }
                acc
            },
            |mut acc, other| {
                for (slot, value) in acc.iter_mut().zip(other) {
                    *slot += value;
                }
                acc
            },
        );
        sums.to_vec()
    }

    /// Reads every column at the bound bit point, one field element per row.
    fn bind_bits(&mut self) {
        let _span = info_span!("bind bits").entered();
        let width = 1 << self.bit_variables;
        let prefix = Point::new(self.bound.as_slice()).equality_weights_msb();
        let fold = WordFold::new(&prefix, width);
        let rows = self.row_equality.len();
        let at_point = |column: &Vec<u64>| -> Vec<F> {
            let mut values = F::zero_vec(rows);
            values[..column.len()]
                .par_iter_mut()
                .zip(column.par_iter())
                .for_each(|(value, &word)| *value = fold.at(word, 0));
            values
        };
        let dot = |table: &[F]| -> F { prefix.iter().zip(table).map(|(&w, &v)| w * v).sum() };
        let p = &self.powers;
        let equality = dot(&self.bit_equality);
        let on_host = self.device.is_none();
        let mut columns = Vec::new();
        let mut weights = Vec::new();
        if on_host {
            columns.push(at_point(&self.words.linear));
            columns.extend(self.words.bitwise.iter().map(at_point));
            weights.push(core::mem::take(&mut self.row_equality));
        }
        let mut scales = [F::ZERO; 7];
        scales[0] = equality;
        scales[1] = equality * p[1];
        if let (Some(product), Some(words)) = (self.product.take(), &self.words.product) {
            if on_host {
                columns.extend(words.iter().map(at_point));
                weights.extend(product.rows);
            }
            let [low, left, right, limb] = product.bits.each_ref().map(|bits| dot(bits));
            scales[2] = p[2] * low;
            scales[3] = p[3] * left;
            scales[4] = p[4] * right;
            scales[5] = p[5] * limb;
            scales[6] = p[6] * limb;
        }
        if let Some(device) = &mut self.device {
            device.bind_bits(&slice_to_device(&prefix), width);
        }
        self.rows = Some(RowTables {
            weights,
            columns,
            scales,
        });
    }
}

/// Adds `weight` times the product of two lines, as a quadratic's coefficients.
fn accumulate_product<F: Field>(acc: &mut [F], weight: F, left: [F; 2], right: [F; 2]) {
    let (dl, dr) = (left[1] - left[0], right[1] - right[0]);
    acc[0] += weight * (left[0] * right[0]);
    acc[1] += weight * (left[0] * dr + right[0] * dl);
    acc[2] += weight * (dl * dr);
}

/// The batched polynomial at one row once every bit is bound.
///
/// `weights` holds the vanishing row weight then the three claim row weights, and `columns`
/// the eight operands.
fn row_value<F: Field>(weights: &[F; 4], columns: &[F; OPERAND_EVALUATIONS], scales: &[F; 7]) -> F {
    let [linear, left, right, output, a, b, low, high] = *columns;
    let inner =
        scales[0] * linear + scales[1] * (left * right - output) + scales[2] * (a * b - low);
    weights[0] * inner
        + scales[3] * weights[1] * a
        + scales[4] * weights[2] * b
        + weights[3] * (scales[5] * low + scales[6] * high)
}

/// Sums every linear term over the rows into one table per bit.
fn linear_sums<F: Field>(
    words: &RelationWords,
    row_equality: &[F],
    product: Option<&ProductWeightTables<F>>,
    width: usize,
) -> [Vec<F>; SUMS] {
    let bytes = width.div_ceil(8);
    let mut terms: Vec<(&[u64], &[F])> = vec![
        (&words.linear, row_equality),
        (&words.bitwise[2], row_equality),
    ];
    if let (Some(product), Some([a, b, low, high])) = (product, &words.product) {
        let [left, right, limb] = &product.rows;
        terms.extend([
            (low.as_slice(), row_equality),
            (a.as_slice(), left.as_slice()),
            (b.as_slice(), right.as_slice()),
            (low.as_slice(), limb.as_slice()),
            (high.as_slice(), limb.as_slice()),
        ]);
    }

    // Each row adds its weight to the slot its byte selects, one table per byte position.
    let rows = terms
        .iter()
        .map(|(column, _)| column.len())
        .max()
        .unwrap_or(0);
    let slots = bytes * 256;
    let accumulated = (0..rows.div_ceil(BLOCK)).into_par_iter().par_fold_reduce(
        || F::zero_vec(terms.len() * slots),
        |mut acc, block| {
            let start = block * BLOCK;
            for (term, (column, weights)) in terms.iter().enumerate() {
                for row in start..column.len().min(start + BLOCK) {
                    let mut value = word(column, row);
                    if value == 0 {
                        continue;
                    }
                    let weight = weights[row];
                    for byte in 0..bytes {
                        let selected = (value & 0xff) as usize;
                        if selected != 0 {
                            acc[term * slots + byte * 256 + selected] += weight;
                        }
                        value >>= 8;
                    }
                }
            }
            acc
        },
        add_into,
    );

    // A bit's sum collects every byte value that sets it.
    array::from_fn(|term| {
        if term >= terms.len() {
            return F::zero_vec(width);
        }
        (0..width)
            .map(|bit| {
                let table = &accumulated[term * slots + (bit / 8) * 256..][..256];
                (0..256)
                    .filter(|selected| selected >> (bit % 8) & 1 == 1)
                    .map(|selected| table[selected])
                    .sum()
            })
            .collect()
    })
}

impl<F: Field> RoundProver<F> for RelationZerocheck<F> {
    fn fold(&mut self, challenge: F) {
        if let (Some(_), Some(device)) = (&self.rows, &mut self.device) {
            let _span = info_span!("row fold").entered();
            device.fold(to_device(challenge).expect("only the device's field reaches the device"));
            return;
        }
        if let Some(tables) = &mut self.rows {
            let _span = info_span!("row fold").entered();
            // Row variables bind most significant first.
            for table in tables.weights.iter_mut().chain(tables.columns.iter_mut()) {
                let half = table.len() / 2;
                let (low, high) = table.split_at_mut(half);
                low.par_iter_mut()
                    .zip(high.par_iter())
                    .for_each(|(low, &high)| *low += challenge * (high - *low));
                table.truncate(half);
            }
            return;
        }
        self.bound.push(challenge);
        if self.bound.len() == self.bit_variables {
            self.bind_bits();
        }
    }

    fn round_poly(&self) -> Vec<F> {
        // The omitted value at one is reconstructed from the running claim.
        match (&self.rows, &self.device) {
            (Some(tables), Some(device)) => {
                let _span = info_span!("row round").entered();
                let scales = slice_to_device(&tables.scales)
                    .try_into()
                    .expect("seven scales");
                let nodes = slice_to_device(&nodes::<F>())
                    .try_into()
                    .expect("three nodes");
                device
                    .row_round(&scales, nodes)
                    .into_iter()
                    .map(from_device)
                    .collect()
            }
            (Some(tables), None) => Self::row_round(tables),
            (None, _) => self.bit_round(),
        }
    }
}

/// Reorders the challenges, bits bound first, into the rows-then-bits point the weights read.
pub(super) fn relation_point<F: Field>(challenges: &Point<F>, bit_variables: usize) -> Point<F> {
    let (bits, rows) = challenges.split_at(bit_variables);
    Point::new(rows.iter().chain(bits.iter()).copied().collect())
}

/// Returns `1, beta, beta^2, ..., beta^6`.
pub(super) fn batching_powers<F: Field>(batching: F) -> [F; PRODUCT_TERMS] {
    let mut powers = [F::ONE; PRODUCT_TERMS];
    for index in 1..PRODUCT_TERMS {
        powers[index] = powers[index - 1] * batching;
    }
    powers
}

/// Evaluates the batched relation polynomial at one point of the cube or its extension.
///
/// ```text
///     eq_zeta * (linear + beta * (left * right - output))
///       + beta^2 * eq_low   * (a * b - lo)
///       + beta^3 * eq_left  * a
///       + beta^4 * eq_right * b
///       + eq_limb * (beta^5 * lo + beta^6 * hi)
/// ```
fn batched<F: Field>(
    equality: F,
    weights: &ProductWeights<F>,
    operands: &[F; OPERAND_EVALUATIONS],
    powers: &[F; PRODUCT_TERMS],
) -> F {
    let [linear, left, right, output, a, b, low, high] = *operands;
    let [low_bit, left_claim, right_claim, limb_claim] = *weights;
    // The five outer terms share one reduction, and so do the two limbs.
    let limbs = F::dot_product::<2>(&[powers[5], powers[6]], &[low, high]);
    F::dot_product::<5>(
        &[
            equality,
            powers[2] * low_bit,
            powers[3] * left_claim,
            powers[4] * right_claim,
            limb_claim,
        ],
        &[
            linear + powers[1] * (left * right - output),
            a * b - low,
            a,
            b,
            limbs,
        ],
    )
}

/// Returns the sum the batched check must reach, fixed by the multiplication claims.
///
/// The local terms vanish on the cube, and each claim term sums to its claimed value.
pub(super) fn claimed_sum<F: Field>(claims: &[F; 4], batching: F) -> F {
    let powers = batching_powers(batching);
    F::dot_product::<4>(&[powers[3], powers[4], powers[5], powers[6]], claims)
}

/// Closes the vanishing check against the operand evaluations the prover supplied.
pub(super) fn closing_value<F: Field>(
    equality: F,
    weights: &ProductWeights<F>,
    operands: &[F; OPERAND_EVALUATIONS],
    batching: F,
) -> F {
    // The verifier evaluates every equality weight itself, so only the operands are claimed.
    batched(equality, weights, operands, &batching_powers(batching))
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryChallenger, BinaryField128, TowerLevel};
    use p3_challenger::HashChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;

    use super::*;

    type F = BinaryField128;

    const BITS: usize = 6;

    fn challenger() -> BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>> {
        BinaryChallenger::from_hasher(Vec::new(), Keccak256Hash)
    }

    // A deterministic stream of distinct words and field elements.
    struct Stream(u64);

    impl Stream {
        fn word(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(0x632b_e59b);
            self.0 ^ (self.0 >> 29)
        }

        fn words(&mut self, len: usize) -> Vec<u64> {
            (0..len).map(|_| self.word()).collect()
        }

        fn point(&mut self, len: usize) -> Vec<F> {
            (0..len)
                .map(|_| F::from_repr(u128::from(self.word()) << 64 | u128::from(self.word())))
                .collect()
        }
    }

    // The bit multilinear of a word column over rows then bits.
    fn dense(column: &[u64], rows: usize) -> Vec<F> {
        let mut table = F::zero_vec(rows << BITS);
        for (row, &word) in column.iter().enumerate() {
            for bit in 0..1 << BITS {
                if word >> bit & 1 == 1 {
                    table[(row << BITS) + bit] = F::ONE;
                }
            }
        }
        table
    }

    fn evaluate(table: &[F], point: &[F]) -> F {
        let weights = Point::new(point).equality_weights_msb();
        table.iter().zip(weights).map(|(&v, w)| v * w).sum()
    }

    // Runs the prover on `words` and checks the verifier's closing formula and operand values.
    fn closes(words: RelationWords, rows_log: usize, stream: &mut Stream) {
        let rows = 1 << rows_log;
        let variables = rows_log + BITS;
        let batching = stream.point(1)[0];
        let vanishing = stream.point(variables);
        let points = words.product.as_ref().map(|_| {
            // The low bit reads every row at bit zero under the vanishing row point.
            let mut low = vanishing[..rows_log].to_vec();
            low.resize(variables, F::ZERO);
            [
                low,
                stream.point(variables),
                stream.point(variables),
                stream.point(variables),
            ]
        });

        // The claimed sum, cell by cell from the batched definition.
        let mut columns = vec![dense(&words.linear, rows)];
        columns.extend(words.bitwise.iter().map(|column| dense(column, rows)));
        if let Some(product) = &words.product {
            columns.extend(product.iter().map(|column| dense(column, rows)));
        }
        let equality = Point::new(vanishing.as_slice()).equality_weights_msb();
        let weights = points.as_ref().map(|points| {
            points
                .each_ref()
                .map(|p| Point::new(p.as_slice()).equality_weights_msb())
        });
        let powers = batching_powers(batching);
        let sum = (0..rows << BITS)
            .map(|cell| {
                let mut operands = [F::ZERO; OPERAND_EVALUATIONS];
                for (slot, column) in operands.iter_mut().zip(&columns) {
                    *slot = column[cell];
                }
                let cell_weights = weights
                    .as_ref()
                    .map_or([F::ZERO; 4], |tables| tables.each_ref().map(|t| t[cell]));
                batched(equality[cell], &cell_weights, &operands, &powers)
            })
            .sum::<F>();

        let mut prover = RelationZerocheck::new(
            &vanishing,
            rows_log,
            BITS,
            words,
            points.clone(),
            batching,
            None,
            false,
        );
        let (proof, _) =
            prover.prove::<F, _>(&mut challenger(), variables, ZEROCHECK_DEGREE, 0, sum);
        let operands = prover.terminal_operands();

        let (end, final_claim) = proof
            .verify(&mut challenger(), variables, ZEROCHECK_DEGREE, 0)
            .unwrap();
        let point = relation_point(&end, BITS);
        let weights_at = points.as_ref().map_or([F::ZERO; 4], |points| {
            points
                .each_ref()
                .map(|p| Point::eval_eq(p, point.as_slice()))
        });
        assert_eq!(
            final_claim,
            closing_value(
                Point::eval_eq(&vanishing, point.as_slice()),
                &weights_at,
                &operands,
                batching
            )
        );
        for (operand, column) in operands.iter().zip(&columns) {
            assert_eq!(*operand, evaluate(column, point.as_slice()));
        }
    }

    #[test]
    fn a_bitwise_zerocheck_closes_on_the_verifier_formula() {
        let mut stream = Stream(3);
        let words = RelationWords {
            linear: stream.words(5),
            bitwise: [stream.words(8), stream.words(7), stream.words(6)],
            product: None,
        };
        closes(words, 3, &mut stream);
    }

    #[test]
    fn a_product_zerocheck_closes_on_the_verifier_formula() {
        let mut stream = Stream(5);
        let words = RelationWords {
            linear: stream.words(3),
            bitwise: [stream.words(4), stream.words(4), stream.words(2)],
            product: Some([
                stream.words(3),
                stream.words(3),
                stream.words(2),
                stream.words(3),
            ]),
        };
        closes(words, 2, &mut stream);
    }

    #[test]
    fn a_single_row_zerocheck_binds_only_bits() {
        let mut stream = Stream(7);
        let words = RelationWords {
            linear: stream.words(1),
            bitwise: [stream.words(1), stream.words(1), stream.words(1)],
            product: Some([
                stream.words(1),
                stream.words(1),
                stream.words(1),
                stream.words(1),
            ]),
        };
        closes(words, 0, &mut stream);
    }
}
