//! The device a word proof's dense work runs on: the product trees and the selector of the
//! multiplication reduction, and the relation vanishing check.
//!
//! A prover on a device hands it to
//! [`WordProofKey::prove_with_hints_on`](crate::WordProofKey::prove_with_hints_on), which then
//! does every one of those steps there.
//!
//! The contract is exact equality with the host, every value a tower-basis bit pattern of
//! `GF(2^128)`.
//!
//! - A tree's leaves are addressed `(row, cell)`, `cells` per row, and every layer above
//!   multiplies adjacent pairs, so the top layer holds one product per row.
//! - The layer sumcheck below layer `d` sums `eq(r, z) * V(z, 0) * V(z, 1)` over the layer
//!   `d + 1` values, binding its most significant variable first.
//! - The selector sumcheck sums `scale_i * eq(point_i, x) * (1 + bit_i(x) * (base(x) - 1))`
//!   over the rows and every bit `i`, sent at the nodes zero, two and three.
//! - The relation vanishing check keeps its word columns and row weights on the device, binds
//!   the bits of every word first and the rows after them, most significant first.
//! - The shift reduction keeps the committed words on the device: a key's coefficient is its
//!   family's weight times the sum, over its references, of the operand's weight and the
//!   equality weight of the reference's relation row; the word phase binds the most
//!   significant word variable first, sent at the nodes zero and two.
//! - The port reduction multiplies two trees over the committed words, cell `c` the word at
//!   index `c`: the identity fingerprints `offset + label * L(c) + v_c` and the permuted ones,
//!   `L(sigma(c))` in place of `L(c)`. Every layer above multiplies adjacent pairs, and its
//!   layer rounds are the product reduction's.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::any::Any;

use p3_binary_field::{BinaryField128, TowerLevel};
use p3_field::Field;

/// The operation columns in the relation check's order: the linear operand, the bitwise
/// left, right and output, then the product left, right, low and high.
pub const OPERATION_COLUMNS: usize = 8;

/// The first product column among the [`OPERATION_COLUMNS`].
pub const PRODUCT_COLUMNS: usize = 4;

/// The linear relation's column among the [`OPERATION_COLUMNS`], as a family mask bit.
pub const LINEAR_FAMILY: u32 = 1;

/// The bitwise relation's three columns, as a family mask bit.
pub const BITWISE_FAMILY: u32 = 2;

/// The product relation's four columns from [`PRODUCT_COLUMNS`], as a family mask bit.
pub const PRODUCT_FAMILY: u32 = 4;

/// A word column a reduction reads: one on the host, or one the device holds.
#[derive(Clone, Copy, Debug)]
pub enum Words<'a> {
    /// Words on the host, zero past their end.
    Host(&'a [u64]),
    /// Operation column `i` the device evaluated, zero past its relations.
    Column(usize),
    /// The committed words the device evaluated, handed over to the reader.
    Witness,
    /// The committed words the device evaluated, lent to the reader and kept after it.
    Held,
}

impl<'a> Words<'a> {
    /// The words, when they are on the host.
    ///
    /// # Panics
    ///
    /// When the device holds them.
    #[must_use]
    pub fn host(self) -> &'a [u64] {
        match self {
            Self::Host(words) => words,
            Self::Column(_) | Self::Witness | Self::Held => panic!("the words are on the device"),
        }
    }
}

/// A device that builds and reduces product trees.
pub trait WordDevice: Send + Sync {
    /// Evaluates the [`OPERATION_COLUMNS`] of the relation families in the mask `families`
    /// from the committed words it holds and `public`, `rows` rows each, every word moved by
    /// its keys' sequences and placed as `witness` and `public_layout` wire it; false, and the
    /// host evaluates them, when it holds no committed words of `witness_len` words. An empty
    /// mask only asks whether it holds them.
    ///
    /// Keys that wire alike share `key`, as in [`Self::shift`].
    #[allow(
        clippy::too_many_arguments,
        reason = "the evaluation's inputs, each one"
    )]
    fn operation_columns(
        &self,
        witness_len: usize,
        public: &[u64],
        key: u64,
        witness: &dyn Fn() -> ShiftLayout,
        public_layout: &dyn Fn() -> ShiftLayout,
        rows: usize,
        families: u32,
    ) -> bool {
        let _ = (witness_len, public, key, witness, public_layout, rows, families);
        false
    }

    /// The power tree over `rows` rows: cell `i` of row `r` is `base_r^(2^i)` when bit `i` of
    /// `right[r]` is set, and one otherwise, `base_r` the product of `lift_powers[t]` over the
    /// set bits `t` of `left[r]`, `width` cells per row, `depth` layers above the leaves.
    #[allow(
        clippy::too_many_arguments,
        reason = "the tree's inputs, each one"
    )]
    fn power_tree(
        &self,
        lift_powers: &[u128],
        left: Words<'_>,
        right: Words<'_>,
        rows: usize,
        width: usize,
        depth: usize,
    ) -> Box<dyn DeviceTree>;

    /// The bit tree over `rows` rows: cell `c * width + t` of row `r` is
    /// `powers[c * width + t]` when bit `t` of `columns[c][r]` is set, and one otherwise,
    /// `depth` layers above the leaves.
    fn bit_tree(
        &self,
        columns: &[Words<'_>],
        powers: &[u128],
        rows: usize,
        width: usize,
        depth: usize,
    ) -> Box<dyn DeviceTree>;

    /// Per column and bit `t`, entry `c * width + t`: the sum over the rows of
    /// `eq(point, r)` over the rows whose word of `columns[c]` has bit `t` set.
    fn bit_values(&self, columns: &[Words<'_>], width: usize, point: &[u128]) -> Vec<u128>;

    /// The selector over `rows` rows, the bits of `right` and the lifts of `left` by
    /// `lift_powers`, bit `i` weighted by `scales[i] * eq(points[i], x)`, its round
    /// polynomials sent at `nodes`.
    #[allow(
        clippy::too_many_arguments,
        reason = "the selector's inputs, each one"
    )]
    fn selector(
        &self,
        points: &[Vec<u128>],
        scales: &[u128],
        right: Words<'_>,
        lift_powers: &[u128],
        left: Words<'_>,
        rows: usize,
        nodes: [u128; 3],
    ) -> Box<dyn DeviceSelector>;

    /// The relation vanishing check over `rows` rows: `columns` are the linear, bitwise left,
    /// right and output words, then the two factors and both limbs when products are declared;
    /// `row_points` the vanishing point's row coordinates, then the left, right and limb claims'.
    ///
    /// Columns the device holds pass to the check, which lets them go.
    fn relation(
        &self,
        columns: &[Words<'_>],
        row_points: &[Vec<u128>],
        rows: usize,
    ) -> Box<dyn DeviceRelation>;

    /// The shift reduction over the committed words `witness`, `witness_len` words of `width`
    /// bits each, wired by `layout`, under the family weights `operation`, the operand weights
    /// `operand`, and the equality weights of `constraint_point`, its first coordinate the
    /// rows' leading bit.
    ///
    /// Keys that wire alike share `key`, so a device may keep a key's wiring and call
    /// `layout` only for a key it does not hold. Committed words the device holds pass to the
    /// reduction, which lets them go.
    #[allow(
        clippy::too_many_arguments,
        reason = "the reduction's inputs, each one"
    )]
    fn shift(
        &self,
        witness: Words<'_>,
        witness_len: usize,
        width: usize,
        key: u64,
        layout: &dyn Fn() -> ShiftLayout,
        operation: &[u128],
        operand: &[u128],
        constraint_point: &[u128],
    ) -> Box<dyn DeviceShift>;

    /// Replaces the committed words it holds with the words at `of`, in that order, so a
    /// statement whose words are a rearrangement of a run's is proven from the run's words in
    /// place; false when it holds no committed words.
    fn gather_witness(&self, of: &[u32]) -> bool;

    /// The port reduction's trees over the committed words it holds, `witness_len` of them,
    /// zero past them up to `2^word_variables` cells: `values` the byte tables reading a word
    /// at the within-word point (256 entries per byte, lowest byte first), `labels` the label
    /// tables of a cell index (256 entries per byte, lowest first), `moved` each moved cell
    /// then its successor. The trees read the committed words where the device holds them, the
    /// opening reading them after; `None` when it holds no committed words of `witness_len`
    /// words.
    #[allow(
        clippy::too_many_arguments,
        reason = "the reduction's inputs, each one"
    )]
    fn port_trees(
        &self,
        witness_len: usize,
        word_variables: usize,
        values: &[u128],
        labels: &[u128],
        offset: u128,
        label: u128,
        moved: &[u32],
    ) -> Option<Box<dyn DevicePorts>>;
}

/// A port reduction's two product trees resident on the device, the identity's first, with at
/// most one layer of the product reduction in progress.
pub trait DevicePorts: Send {
    /// Each tree's product.
    fn roots(&self) -> [u128; 2];

    /// The first two values of each tree's layer `depth` above the cells.
    fn binary_children(&self, depth: usize) -> [[u128; 2]; 2];

    /// Starts the radix-four layer whose children are each tree's layer `depth`, at `point`:
    /// row `i` of four children weighs `eq(point, i)`, coordinate `k` binding bit `k` of `i`.
    fn begin_layer(&mut self, depth: usize, point: &[u128]);

    /// The current round's message at the five `nodes`: the trees' rows' products, the trees
    /// combined by the powers of `batching`, under the equality weights.
    fn round(&self, nodes: &[u128; 5], batching: u128) -> [u128; 5];

    /// Binds the current round's variable, the lowest bit of the row, to `challenge`.
    fn fold(&mut self, challenge: u128);

    /// Each tree's four children once every round of the layer is folded.
    fn children(&self) -> [[u128; 4]; 2];

    /// At `point`, its first coordinate the cells' leading bit: the cells' values' extension,
    /// and the sum over the moved cells of `eq(point, c) * (L(sigma(c)) - L(c))`.
    fn sums(&self, point: &[u128]) -> [u128; 2];
}

/// A committed segment's compiled shift wiring, flattened.
#[derive(Clone, Debug, Default)]
pub struct ShiftLayout {
    /// Ten words per instance block: its first word, words per instance, instances, first
    /// slot, then each family's relations per instance, then each family's first relation.
    /// Bit 31 of the instances word marks a block whose words are interleaved, bit 30 one whose
    /// relations are: a power-of-two instance count whose index is their low bits.
    pub blocks: Vec<u32>,
    /// Two words per slot: its first key, and the key past its last.
    pub slot_keys: Vec<u32>,
    /// Four words per key: its family's code, its shift sequence, its first reference, and
    /// the reference past its last.
    pub keys: Vec<u32>,
    /// Each reference's operand position in the batching cube.
    pub operands: Vec<u8>,
    /// Each reference's relation position within its instance.
    pub constraints: Vec<u32>,
    /// The distinct shift sequences.
    pub shifts: usize,
    /// Four bytes per shift sequence: the inner movement's kind and distance, then the
    /// outer's, the kind numbered as [`shift_kind_code`].
    pub sequences: Vec<u8>,
}

/// A shift kind's number on a device: logical left, logical right, arithmetic right and rotate
/// right over the whole word, then the same four over each 32-bit lane.
#[must_use]
pub const fn shift_kind_code(kind: p3_word::ShiftKind) -> u8 {
    use p3_word::ShiftKind;
    match kind {
        ShiftKind::LogicalLeft => 0,
        ShiftKind::LogicalRight => 1,
        ShiftKind::ArithmeticRight => 2,
        ShiftKind::RotateRight => 3,
        ShiftKind::Lane32LogicalLeft => 4,
        ShiftKind::Lane32LogicalRight => 5,
        ShiftKind::Lane32ArithmeticRight => 6,
        ShiftKind::Lane32RotateRight => 7,
    }
}

/// A shift reduction resident on the device.
pub trait DeviceShift: Send {
    /// The bit phase's witness rows, `width` per shift sequence: entry `s * width + b` sums
    /// the coefficients of the keys with sequence `s` over the words whose bit `b` is set.
    fn bit_rows(&self) -> Vec<u128>;

    /// Starts the word phase over `num_variables` word variables: one factor the words read
    /// through the byte tables `bytes` (256 entries per byte, lowest byte first), the other
    /// each word's keys' coefficients times `shift_evaluations` of their sequences.
    fn word_phase(
        &mut self,
        bytes: &[u128],
        shift_evaluations: &[u128],
        num_variables: usize,
        two: u128,
    );

    /// The word round's values at zero and two.
    fn round(&self) -> [u128; 2];

    /// Binds the most significant word variable to `challenge`.
    fn fold(&mut self, challenge: u128);

    /// Both factors once every word variable is bound.
    fn terminal(&self) -> [u128; 2];
}

/// A relation vanishing check resident on the device.
///
/// A word folds over a bound prefix `p` of its bits as `sum_i eq(p, i) * bit(i * stride + s)`
/// at position `s`, `stride` being its width over the prefix's length.
pub trait DeviceRelation: Send + Sync {
    /// Every linear term summed over the rows, `width` bits each, in the host's term order: the
    /// linear and output words and the low limb under the vanishing weights, then the left
    /// factor, right factor, low and high limbs under their claims' weights.
    fn linear_sums(&self, width: usize) -> Vec<u128>;

    /// The bitwise products folded over the prefix weights `prefix`, as the coefficients of a
    /// quadratic per remaining position under the vanishing weights, then the low bit's when
    /// products are declared.
    fn bit_products(&self, prefix: &[u128], width: usize) -> Vec<u128>;

    /// Reads every column at the full bit point whose equality weights are `prefix`.
    fn bind_bits(&mut self, prefix: &[u128], width: usize);

    /// The row round polynomial at `nodes` under the batching `scales`.
    fn row_round(&self, scales: &[u128; 7], nodes: [u128; 3]) -> [u128; 3];

    /// Binds the most significant row variable to `challenge`.
    fn fold(&mut self, challenge: u128);

    /// Every column once every variable is bound.
    fn terminal(&self) -> Vec<u128>;
}

/// A selector sumcheck resident on the device.
pub trait DeviceSelector: Send {
    /// The current round polynomial at the nodes.
    fn round(&self) -> [u128; 3];

    /// Binds the most significant variable to `challenge`.
    fn fold(&mut self, challenge: u128);

    /// The base lift, then every bit, once every variable is bound.
    fn terminal(&self) -> Vec<u128>;
}

/// A product tree resident on the device, with at most one layer sumcheck in progress.
pub trait DeviceTree: Send {
    /// The top layer, one product per row, evaluated at `point`, its first coordinate the
    /// rows' leading bit.
    fn top_at(&self, point: &[u128]) -> u128;

    /// Starts the sumcheck below layer `layer`, at `point`, over the layer under it.
    fn begin(&mut self, layer: usize, point: &[u128]);

    /// Sums over the cube past the bound variable, under the equality weights of its point.
    ///
    /// ```text
    ///     [V(0, rest, 0) V(0, rest, 1),  (V(0, rest, 0) + V(1, rest, 0)) (V(0, rest, 1) + V(1, rest, 1)),
    ///      V(1, rest, 0) V(1, rest, 1)]
    /// ```
    ///
    /// The last, which the round's claim fixes unless its weights vanish, is zero unless
    /// `at_one`.
    fn sums(&self, at_one: bool) -> [u128; 3];

    /// Binds the most significant variable to `challenge`; the next round's sums need the
    /// one at one when `next_at_one`.
    fn fold(&mut self, challenge: u128, next_at_one: bool);

    /// Both halves once every variable is bound.
    fn halves(&self) -> [u128; 2];

    /// Each leaf cell's row extension at `point`, read from the leaves as built.
    fn leaf_values(&self, point: &[u128]) -> Vec<u128>;
}

/// Whether `F` is the device's field.
pub(crate) fn is_device_field<F: Field>() -> bool {
    to_device(F::ONE).is_some()
}

/// The tower-basis pattern of `value`, when `F` is the device's field.
pub(crate) fn to_device<F: Field>(value: F) -> Option<u128> {
    (&value as &dyn Any)
        .downcast_ref::<BinaryField128>()
        .map(|value| value.to_repr())
}

/// The element of `F` a tower-basis pattern names; `F` is the device's field.
pub(crate) fn from_device<F: Field>(value: u128) -> F {
    *(&BinaryField128::from_repr(value) as &dyn Any)
        .downcast_ref::<F>()
        .expect("only the device's field reaches the device")
}

/// The patterns of every element of `values`; `F` is the device's field.
pub(crate) fn slice_to_device<F: Field>(values: &[F]) -> Vec<u128> {
    values
        .iter()
        .map(|&value| to_device(value).expect("only the device's field reaches the device"))
        .collect()
}
