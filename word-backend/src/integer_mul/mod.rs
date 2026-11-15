//! Full-width unsigned multiplication, reduced to evaluation claims on its four operand columns.
//!
//! Each multiplication row states one integer identity over words of width `w = 2^k`.
//!
//! ```text
//!     a * b = lo + 2^w * hi        (as integers)
//! ```
//!
//! A binary field has no integer arithmetic, so both sides are lifted into an exponent.
//!
//! ```text
//!     g         a generator of the challenge field's multiplicative group, of order 2^n - 1
//!     G_t       g^(2^t)
//!
//!     base(x)   = prod_{i < w}  (1 + a_i(x) * (G_i - 1))                 = g^a
//!     power(x)  = prod_{i < w}  (1 + b_i(x) * (base(x)^(2^i) - 1))       = g^(a * b)
//!     result(x) = prod_{t < 2w} (1 + c_t(x) * (G_t - 1))                 = g^(lo + 2^w * hi)
//! ```
//!
//! The reduction proves `power(row) = result(row)` on every row, then ties both sides to bits.
//!
//! 1. A random row point compares the two lifts at one value, the root.
//! 2. A product check walks the power tree to its `w` leaf columns at one row point.
//! 3. Each leaf column is a Frobenius power of a selector of `base`, so twisting the claims
//!    turns them into claims on `base` and on the bits of `b`, which one sumcheck checks.
//! 4. A product check walks the base tree from that claim on `base` to the bits of `a`.
//! 5. A product check walks the result tree from the root to the bits of both limbs.
//!
//! Squaring is linear over GF(2), so for a Boolean `x` and every bit `i`
//!
//! ```text
//!     phi(y) = y^2,   phi^(-i)(1 + b_i * (base^(2^i) - 1)) = 1 + b_i * (base - 1)
//!     phi^(-i)(sum_x eq(r, x) * v(x)) = sum_x eq(phi^(-i)(r), x) * phi^(-i)(v(x))
//! ```
//!
//! so a claim on leaf column `i` at `r` becomes a claim on `1 + b_i * (base - 1)` at the
//! twisted point, and every bit's claim reads the same `base`.
//!
//! Each tree's prover sends the row extension of every leaf column at the point its layers end
//! on, and a random point folds those per-bit values into one claim on the operand column.
//!
//! # Soundness
//!
//! Equal lifts mean equal exponents modulo `2^n - 1`.
//!
//! ```text
//!     n >= 2w                       checked by both the prover and the verifier
//!     a * b      <= (2^w - 1)^2     <  2^n - 1
//!     lo + 2^w hi <= 2^(2w) - 1    <= 2^n - 1
//! ```
//!
//! Congruent exponents in that range are equal, with one exception.
//!
//! When `n = 2w`, the all-ones limbs `2^n - 1` lift to one, exactly as a zero product does.
//!
//! The caller closes that gap with the low-bit relation `lo_0 = a_0 * b_0`.
//!
//! An odd claimed product then needs two odd factors, whose product is never zero.
//!
//! The statistical error, over a field of `2^n` elements, is a sum of Schwartz-Zippel terms.
//!
//! ```text
//!     row point       m / 2^n
//!     tree of depth D sum_{d < D} (3 * (m + d) + 1) / 2^n
//!     trees           power and base of depth k, result of depth k + 1
//!     selector        3 * m / 2^n
//!     folds           four points of k coordinates, k / 2^n each
//! ```
//!
//! The twist is a bijection, so it adds no error.
//!
//! The field check runs at different moments on the two sides.
//!
//! - The prover refuses the field before it commits to anything.
//! - The verifier refuses it at the start of its multiplication step.
//! - A word proof reaches that step after absorbing the commitment and the statement.
//!
//! Either way a small field never yields an accepted proof.
//!
//! The prover performs `O(2^m * w)` field operations and holds `O(2^m * w)` field elements.
//!
//! ```text
//!     power and base leaves     w per row
//!     result leaves             2w per row
//!     selector tables           2w per row
//! ```

mod error;
mod exponent;
mod record;
mod round;
mod transcript;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::iter;

pub use error::IntegerMulError;
use exponent::{evaluate, generator_squarings};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_security::word::WordIntegerMulSecurityModel;
use p3_sumcheck::generic_degree::RoundProver;
pub use record::IntegerMulProof;
use record::{LayerProof, TreeProof};
use round::{DEGREE, ProductLayer, Selector};
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};
use transcript::{ProverTranscript, TranscriptShape, VerifierTranscript};

use crate::device::{DeviceTree, PRODUCT_COLUMNS, WordDevice, Words, from_device, slice_to_device};
use crate::{Packed, PackedWord};

/// Rows one parallel task covers.
const BLOCK: usize = 1 << 12;

/// One of the three product trees the reduction walks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Tree {
    /// The tree whose leaves select `base^(2^i)` by bit `i` of the right factor.
    Power,
    /// The tree whose leaves select `g^(2^i)` by bit `i` of the left factor.
    Base,
    /// The tree whose leaves select `g^(2^t)` by bit `t` of the two limbs.
    Result,
}

impl Tree {
    /// Returns the number of layers between the root and the leaves.
    const fn depth(self, bit_variables: usize) -> usize {
        match self {
            Self::Power | Self::Base => bit_variables,
            // One index over twice the word width addresses one leaf.
            Self::Result => bit_variables + 1,
        }
    }

    /// Returns the number of leaf columns, one per bit the tree selects by.
    const fn leaves(self, bit_variables: usize) -> usize {
        1 << self.depth(bit_variables)
    }

    /// Returns the name reported by a shape mismatch.
    const fn name(self) -> &'static str {
        match self {
            Self::Power => "power",
            Self::Base => "base",
            Self::Result => "result",
        }
    }

    /// Returns the transcript label around one layer sumcheck.
    const fn layer_label(self) -> &'static str {
        match self {
            Self::Power => "power_layer",
            Self::Base => "base_layer",
            Self::Result => "result_layer",
        }
    }

    /// Returns the transcript label of the leaf evaluations.
    const fn leaves_label(self) -> &'static str {
        match self {
            Self::Power => "power_leaves",
            Self::Base => "base_leaves",
            Self::Result => "result_leaves",
        }
    }

    /// Returns the transcript label of the point folding the leaf evaluations.
    const fn point_label(self) -> &'static str {
        match self {
            Self::Power => "fold_point",
            Self::Base => "left_point",
            Self::Result => "limb_point",
        }
    }
}

/// One evaluation claim on an operand column over the multiplication rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OperandClaim<F> {
    /// Row coordinates followed by within-word coordinates.
    pub(crate) point: Vec<F>,
    /// Claimed evaluation of the column's bits.
    pub(crate) value: F,
}

/// The multiplication reduction of one statement.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct IntegerMulReduction {
    /// Variables selecting one padded multiplication row.
    row_variables: usize,
    /// Variables selecting one bit within a word.
    bit_variables: usize,
}

impl IntegerMulReduction {
    /// Creates the reduction for a statement, or nothing when it declares no product.
    pub(crate) const fn new(relations: usize, bit_variables: usize) -> Option<Self> {
        if relations == 0 {
            return None;
        }
        Some(Self {
            row_variables: relations.next_power_of_two().ilog2() as usize,
            bit_variables,
        })
    }

    /// Returns the word width.
    const fn width(&self) -> usize {
        1 << self.bit_variables
    }

    /// Rejects a challenge field too small for the lift, or not of characteristic two.
    ///
    /// # Errors
    ///
    /// Returns an error when the field has fewer than `2^(2w)` elements or its order is not a
    /// power of two.
    pub(crate) fn check_field<EF: Field>(&self) -> Result<(), IntegerMulError> {
        // A field of order at least 2^(2w) has a group of order at least 2^(2w) - 1.
        let required = 2 * self.width();
        if EF::order().bits() <= required as u64 {
            return Err(IntegerMulError::FieldTooSmall { required });
        }
        if binary_degree::<EF>().is_none() {
            return Err(IntegerMulError::FieldNotBinary);
        }
        Ok(())
    }

    /// Returns the soundness model of the executed schedule.
    pub(crate) const fn security_model(&self, field_bits: usize) -> WordIntegerMulSecurityModel {
        match WordIntegerMulSecurityModel::new(field_bits, self.row_variables, self.bit_variables) {
            Some(model) => model,
            None => panic!("a nonzero field width gives a valid multiplication model"),
        }
    }

    /// Dimensions bound into the reduction's own transcript.
    const fn transcript_shape(&self) -> TranscriptShape {
        TranscriptShape {
            row_variables: self.row_variables,
            bit_variables: self.bit_variables,
        }
    }

    /// Proves every product and returns claims on the left, right, low, and high columns.
    ///
    /// `columns` are on the host, or `None` when `device` holds them as its product columns.
    ///
    /// The field must already have passed [`Self::check_field`].
    pub(crate) fn prove<F, EF, W, C>(
        &self,
        columns: Option<&[Vec<Packed<W>>; 4]>,
        device: Option<&dyn WordDevice>,
        challenger: &mut C,
    ) -> (IntegerMulProof<F, EF>, [OperandClaim<EF>; 4])
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        W: PackedWord,
        C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Padding rows are zero products, which lift to one on both sides.
        let rows = 1 << self.row_variables;
        let host = columns.map(|columns| {
            columns.each_ref().map(|column| {
                let mut words = column
                    .par_iter()
                    .map(|&packed| W::unpack(packed).to_u64())
                    .collect::<Vec<_>>();
                words.resize(rows, 0);
                words
            })
        });
        let words: [Words<'_>; 4] = match &host {
            Some(host) => core::array::from_fn(|i| Words::Host(&host[i])),
            None => core::array::from_fn(|i| Words::Column(PRODUCT_COLUMNS + i)),
        };
        let mut transcript = ProverTranscript::<C, F, EF>::new(challenger, self.transcript_shape());
        let proved = self.prove_words(&mut transcript, words, [words[0], words[1]], device);
        transcript.finish();
        proved
    }

    /// Runs the reduction over padded words, the power tree built over `power`'s factors, every
    /// tree and the selector on `device` when there is one.
    fn prove_words<C, F, EF>(
        &self,
        transcript: &mut ProverTranscript<'_, C, F, EF>,
        words: [Words<'_>; 4],
        power: [Words<'_>; 2],
        device: Option<&dyn WordDevice>,
    ) -> (IntegerMulProof<F, EF>, [OperandClaim<EF>; 4])
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let (m, k, width) = (self.row_variables, self.bit_variables, self.width());
        let degree = binary_degree::<EF>().expect("the field is binary");
        let powers = generator_squarings::<EF>(2 * width);
        let [left, right, low, high] = words;

        let rows = 1 << m;

        // Power tree: the root compares both lifts, and the leaves leave one value per bit.
        let mut tree = Layers::power(
            &powers[..width],
            power[0],
            power[1],
            rows,
            width,
            Tree::Power.depth(k),
            device,
        );
        let (row_point, root) = transcript.row_point(m, |point| tree.top_at(point));
        let (layers, point, _) =
            prove_layers(transcript, Tree::Power, &mut tree, row_point.clone(), root);
        let leaves = tree.leaf_values(width, &point[..m]);
        drop(tree);
        let fold = transcript.leaves(Tree::Power, &leaves, k);

        // Selector: the twisted leaf claims, batched by the fold, read the base and right bits.
        let gamma = Point::new(fold.as_slice()).equality_weights_msb();
        let claimed = twisted_sum(&leaves, &gamma, degree);
        let twisted = twisted_points(&point[..m], width, degree);
        let selector = device.map(|device| {
            device.selector(
                &twisted
                    .iter()
                    .map(|point| slice_to_device(point))
                    .collect::<Vec<_>>(),
                &slice_to_device(&gamma),
                right,
                &slice_to_device(&powers[..width]),
                left,
                rows,
                Selector::<EF>::device_nodes(),
            )
        });
        let mut prover = selector.map_or_else(
            || {
                let weights = twisted
                    .iter()
                    .zip(&gamma)
                    .map(|(twisted, &scale)| {
                        let mut table = Point::new(twisted.as_slice()).equality_weights_msb();
                        table.par_iter_mut().for_each(|value| *value *= scale);
                        table
                    })
                    .collect();
                let base = lifts(left.host(), &powers[..width]);
                let bits = (0..width)
                    .map(|bit| {
                        right
                            .host()
                            .par_iter()
                            .map(|&word| EF::from_bool((word >> bit) & 1 == 1))
                            .collect()
                    })
                    .collect();
                Selector::new(weights, bits, base)
            },
            Selector::Device,
        );
        let ((selector, end), selected, right_point) = transcript.selector(k, |challenger| {
            let (proof, end) = info_span!("selector")
                .in_scope(|| prover.prove::<F, _>(challenger, m, DEGREE, 0, claimed));
            ((proof, end), prover.terminal())
        });
        drop(prover);
        let end = end.as_slice().to_vec();
        let right_claim = OperandClaim {
            point: [end.as_slice(), &right_point].concat(),
            value: evaluate(&selected[1..], &right_point),
        };
        let power = TreeProof { layers, leaves };

        // Base tree: from the selector's claim on the base lift down to the left bits.
        let mut tree = Layers::bits(
            &[left],
            &powers[..width],
            rows,
            width,
            Tree::Base.depth(k),
            device,
        );
        let (layers, point, _) = prove_layers(transcript, Tree::Base, &mut tree, end, selected[0]);
        drop(tree);
        let leaves = column_bits(&[left], width, &point[..m], device);
        let left_point = transcript.leaves(Tree::Base, &leaves, k);
        let left_claim = OperandClaim {
            point: [&point[..m], &left_point].concat(),
            value: evaluate(&leaves, &left_point),
        };
        let base = TreeProof { layers, leaves };

        // Result tree: from the shared root down to the bits of both limbs.
        let mut tree =
            Layers::bits(&[low, high], &powers, rows, width, Tree::Result.depth(k), device);
        let (layers, point, _) = prove_layers(transcript, Tree::Result, &mut tree, row_point, root);
        drop(tree);
        let leaves = column_bits(&[low, high], width, &point[..m], device);
        let limb_point = transcript.leaves(Tree::Result, &leaves, k);
        let limb_claims = leaves.split_at(width);
        let limb_claim = |bits: &[EF]| OperandClaim {
            point: [&point[..m], &limb_point].concat(),
            value: evaluate(bits, &limb_point),
        };
        let claims = [
            left_claim,
            right_claim,
            limb_claim(limb_claims.0),
            limb_claim(limb_claims.1),
        ];
        let result = TreeProof { layers, leaves };

        (
            IntegerMulProof {
                root,
                power,
                selector,
                selected,
                base,
                result,
            },
            claims,
        )
    }

    /// Verifies every product and returns claims on the left, right, low, and high columns.
    ///
    /// # Errors
    ///
    /// Returns an error when the field is too small, the record is malformed, or a check fails.
    pub(crate) fn verify<F, EF, C>(
        &self,
        proof: &IntegerMulProof<F, EF>,
        challenger: &mut C,
    ) -> Result<[OperandClaim<EF>; 4], IntegerMulError>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // The field and every record shape are checked before the transcript absorbs anything.
        self.check_field::<EF>()?;
        let k = self.bit_variables;
        for (tree, record) in [
            (Tree::Power, &proof.power),
            (Tree::Base, &proof.base),
            (Tree::Result, &proof.result),
        ] {
            let expected = tree.depth(k);
            if record.layers.len() != expected {
                return Err(IntegerMulError::TreeDepth {
                    tree: tree.name(),
                    expected,
                    actual: record.layers.len(),
                });
            }
            let expected = tree.leaves(k);
            if record.leaves.len() != expected {
                return Err(IntegerMulError::LeafCount {
                    step: tree.name(),
                    expected,
                    actual: record.leaves.len(),
                });
            }
        }
        if proof.selected.len() != self.width() + 1 {
            return Err(IntegerMulError::LeafCount {
                step: "selector",
                expected: self.width() + 1,
                actual: proof.selected.len(),
            });
        }

        let mut transcript =
            VerifierTranscript::<C, F, EF>::new(challenger, self.transcript_shape());
        match self.replay(&mut transcript, proof) {
            Ok(claims) => {
                transcript.finish();
                Ok(claims)
            }
            Err(error) => {
                transcript.abort();
                Err(error)
            }
        }
    }

    /// Replays every step, stopping at the first failed check.
    fn replay<C, F, EF>(
        &self,
        transcript: &mut VerifierTranscript<'_, C, F, EF>,
        proof: &IntegerMulProof<F, EF>,
    ) -> Result<[OperandClaim<EF>; 4], IntegerMulError>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let (m, k, width) = (self.row_variables, self.bit_variables, self.width());
        let degree = binary_degree::<EF>().ok_or(IntegerMulError::FieldNotBinary)?;
        let powers = generator_squarings::<EF>(2 * width);
        let malformed = |step, values: &[EF], expected| IntegerMulError::LeafCount {
            step,
            expected,
            actual: values.len(),
        };
        let row_point = transcript.row_point(m, proof.root);

        // Power tree: the leaf values must extend the claim the layers left.
        let (point, claim) = verify_layers(
            transcript,
            Tree::Power,
            &proof.power.layers,
            row_point.clone(),
            proof.root,
        )?;
        let (x, z) = point.split_at(m);
        if claim != evaluate(&proof.power.leaves, z) {
            return Err(IntegerMulError::LeafClaim);
        }
        let fold = transcript
            .leaves(Tree::Power, &proof.power.leaves, k)
            .ok_or_else(|| malformed(Tree::Power.name(), &proof.power.leaves, width))?;

        // Selector: its sum is the twisted leaf values, and it closes on the selected values.
        let gamma = Point::new(fold.as_slice()).equality_weights_msb();
        if proof.selector.claimed_sum != twisted_sum(&proof.power.leaves, &gamma, degree) {
            return Err(IntegerMulError::EnteringClaim);
        }
        let (replay, right_point) = transcript
            .selector(&proof.selected, k, |challenger| {
                proof.selector.verify(challenger, m, DEGREE, 0)
            })
            .ok_or_else(|| malformed("selector", &proof.selected, width + 1))?;
        let (end, value) = replay?;
        let end = end.as_slice();
        let (base, bits) = (proof.selected[0], &proof.selected[1..]);
        let expected: EF = twisted_points(x, width, degree)
            .iter()
            .zip(&gamma)
            .zip(bits)
            .map(|((twisted, &scale), &bit)| {
                scale * Point::eval_eq(twisted, end) * (EF::ONE + bit * (base - EF::ONE))
            })
            .sum();
        if value != expected {
            return Err(IntegerMulError::SelectorClaim);
        }
        let right_claim = OperandClaim {
            point: [end, &right_point].concat(),
            value: evaluate(bits, &right_point),
        };

        // Base tree: from the selected base lift down to the left bits.
        let (point, claim) = verify_layers(
            transcript,
            Tree::Base,
            &proof.base.layers,
            end.to_vec(),
            base,
        )?;
        let (x, z) = point.split_at(m);
        if claim != leaf_closing(&proof.base.leaves, &powers[..width], z) {
            return Err(IntegerMulError::LeafClaim);
        }
        let left_point = transcript
            .leaves(Tree::Base, &proof.base.leaves, k)
            .ok_or_else(|| malformed(Tree::Base.name(), &proof.base.leaves, width))?;
        let left_claim = OperandClaim {
            point: [x, &left_point].concat(),
            value: evaluate(&proof.base.leaves, &left_point),
        };

        // Result tree: from the shared root down to the bits of both limbs.
        let (point, claim) = verify_layers(
            transcript,
            Tree::Result,
            &proof.result.layers,
            row_point,
            proof.root,
        )?;
        let (x, z) = point.split_at(m);
        if claim != leaf_closing(&proof.result.leaves, &powers, z) {
            return Err(IntegerMulError::LeafClaim);
        }
        let limb_point = transcript
            .leaves(Tree::Result, &proof.result.leaves, k)
            .ok_or_else(|| malformed(Tree::Result.name(), &proof.result.leaves, 2 * width))?;
        let (low, high) = proof.result.leaves.split_at(width);
        let limb_claim = |bits: &[EF]| OperandClaim {
            point: [x, &limb_point].concat(),
            value: evaluate(bits, &limb_point),
        };

        Ok([left_claim, right_claim, limb_claim(low), limb_claim(high)])
    }
}

/// Returns the degree of the field over GF(2), or nothing when its order is not a power of two.
fn binary_degree<F: Field>() -> Option<usize> {
    let order = F::order();
    (order.count_ones() == 1).then(|| order.bits() as usize - 1)
}

/// Applies `phi^(-i)`, the inverse of the `i`-th Frobenius power, to one element.
fn twist<F: Field>(value: F, i: usize, degree: usize) -> F {
    // phi has order `degree`, so its inverse power is the complementary forward power.
    value.exp_power_of_2((degree - i % degree) % degree)
}

/// Returns `phi^(-i)(point)`, coordinate by coordinate, for every `i < count`.
fn twisted_points<F: Field>(point: &[F], count: usize, degree: usize) -> Vec<Vec<F>> {
    let squarings = point
        .iter()
        .map(|&coordinate| {
            iter::successors(Some(coordinate), |value| Some(value.square()))
                .take(degree)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    (0..count)
        .map(|i| {
            squarings
                .iter()
                .map(|powers| powers[(degree - i % degree) % degree])
                .collect()
        })
        .collect()
}

/// Returns `sum_i gamma_i * phi^(-i)(leaf_i)`, the selector sumcheck's claimed sum.
fn twisted_sum<F: Field>(leaves: &[F], gamma: &[F], degree: usize) -> F {
    leaves
        .iter()
        .zip(gamma)
        .enumerate()
        .map(|(i, (&leaf, &scale))| scale * twist(leaf, i, degree))
        .sum()
}

/// Returns `sum_t eq(z, t) * (1 + bit_t * (powers_t - 1))`, a selector leaf layer at `z`.
fn leaf_closing<F: Field>(bits: &[F], powers: &[F], z: &[F]) -> F {
    Point::new(z)
        .equality_weights_msb()
        .into_iter()
        .zip(bits)
        .zip(powers)
        .map(|((weight, &bit), &power)| weight * (F::ONE + bit * (power - F::ONE)))
        .sum()
}

/// Adds two equal-length accumulators.
fn add_into<F: Field>(mut acc: Vec<F>, other: Vec<F>) -> Vec<F> {
    for (slot, value) in acc.iter_mut().zip(other) {
        *slot += value;
    }
    acc
}

/// Returns `g^x` for every row's word, from byte tables of the squarings `powers[t] = g^(2^t)`.
#[instrument(skip_all)]
fn lifts<F: Field>(words: &[u64], powers: &[F]) -> Vec<F> {
    let tables = powers
        .chunks(8)
        .map(|group| {
            let mut table = vec![F::ONE; 1 << group.len()];
            for subset in 1..table.len() {
                // Each subset extends the one without its lowest member.
                table[subset] =
                    table[subset & (subset - 1)] * group[subset.trailing_zeros() as usize];
            }
            table
        })
        .collect::<Vec<_>>();
    words
        .par_iter()
        .map(|&word| {
            tables
                .iter()
                .enumerate()
                .map(|(group, table)| table[(word >> (8 * group)) as usize & (table.len() - 1)])
                .product()
        })
        .collect()
}

/// Returns the power-tree leaves of every row, addressed `(row, i)`.
///
/// Leaf `i` is `base^(2^i)` when bit `i` of the right factor is set, and one otherwise.
#[instrument(skip_all)]
fn power_leaves<F: Field>(bases: &[F], right: &[u64], width: usize) -> Vec<F> {
    let mut leaves = F::zero_vec(bases.len() * width);
    leaves
        .par_chunks_mut(width)
        .zip(bases.par_iter().zip(right.par_iter()))
        .for_each(|(row, (&base, &word))| {
            let mut power = base;
            for (i, leaf) in row.iter_mut().enumerate() {
                *leaf = if (word >> i) & 1 == 1 { power } else { F::ONE };
                power = power.square();
            }
        });
    leaves
}

/// Returns the leaves selecting `powers[t]` by bit `t` of each row's words, addressed `(row, t)`.
///
/// The words of one row are read in order, `width` bits each.
#[instrument(skip_all)]
fn bit_leaves<F: Field>(columns: &[&[u64]], powers: &[F], width: usize) -> Vec<F> {
    let cells = columns.len() * width;
    let mut leaves = vec![F::ONE; columns[0].len() * cells];
    leaves
        .par_chunks_mut(cells)
        .enumerate()
        .for_each(|(row, leaves)| {
            for (limb, column) in columns.iter().enumerate() {
                let word = column[row];
                for bit in 0..width {
                    if (word >> bit) & 1 == 1 {
                        leaves[limb * width + bit] = powers[limb * width + bit];
                    }
                }
            }
        });
    leaves
}

/// Returns each leaf column's row extension at `point`.
#[instrument(skip_all)]
fn leaf_values<F: Field>(leaves: &[F], count: usize, point: &[F]) -> Vec<F> {
    let weights = Point::new(point).equality_weights_msb();
    leaves
        .par_chunks(count)
        .zip(weights.par_iter())
        .par_fold_reduce(
            || F::zero_vec(count),
            |mut acc, (row, &weight)| {
                for (slot, &value) in acc.iter_mut().zip(row) {
                    *slot += weight * value;
                }
                acc
            },
            add_into,
        )
}

/// Returns each bit's row extension at `point`, word after word, `width` bits each, on
/// `device` when there is one.
#[instrument(skip_all, name = "bit_values")]
fn column_bits<F: Field>(
    columns: &[Words<'_>],
    width: usize,
    point: &[F],
    device: Option<&dyn WordDevice>,
) -> Vec<F> {
    match device {
        Some(device) => device
            .bit_values(columns, width, &slice_to_device(point))
            .into_iter()
            .map(from_device)
            .collect(),
        None => {
            let columns = columns.iter().map(|words| words.host()).collect::<Vec<_>>();
            bit_values(&columns, width, point)
        }
    }
}

/// Returns each bit's row extension at `point`, word after word, `width` bits each.
fn bit_values<F: Field>(columns: &[&[u64]], width: usize, point: &[F]) -> Vec<F> {
    let weights = Point::new(point).equality_weights_msb();
    let bytes = width.div_ceil(8);
    let rows = weights.len();

    // Each row adds its weight to the slot its byte selects, one table per byte position.
    let slots = columns.len() * bytes * 256;
    let accumulated = (0..rows.div_ceil(BLOCK)).into_par_iter().par_fold_reduce(
        || F::zero_vec(slots),
        |mut acc, block| {
            for row in block * BLOCK..rows.min((block + 1) * BLOCK) {
                let weight = weights[row];
                for (index, column) in columns.iter().enumerate() {
                    let mut word = column[row];
                    for byte in 0..bytes {
                        let selected = (word & 0xff) as usize;
                        if selected != 0 {
                            acc[(index * bytes + byte) * 256 + selected] += weight;
                        }
                        word >>= 8;
                    }
                }
            }
            acc
        },
        add_into,
    );

    // A bit's value collects every byte value that sets it.
    (0..columns.len())
        .flat_map(|index| {
            let accumulated = &accumulated;
            (0..width).map(move |bit| {
                let table = &accumulated[(index * bytes + bit / 8) * 256..][..256];
                (0..256)
                    .filter(|selected| (selected >> (bit % 8)) & 1 == 1)
                    .map(|selected| table[selected])
                    .sum()
            })
        })
        .collect()
}

/// Builds every layer of a product tree from its leaves, root first.
///
/// ```text
///     V_d(z) = V_{d+1}(z, 0) * V_{d+1}(z, 1)
/// ```
#[instrument(skip_all)]
fn product_layers<F: Field>(leaves: Vec<F>, depth: usize) -> Vec<Vec<F>> {
    let mut layers = Vec::with_capacity(depth + 1);
    layers.push(leaves);
    for _ in 0..depth {
        let below = layers.last().expect("the leaves are always present");
        let above = below
            .par_chunks_exact(2)
            .map(|pair| pair[0] * pair[1])
            .collect();
        layers.push(above);
    }
    layers.reverse();
    layers
}

/// Every layer of one product tree, root first, on the host or on the device.
enum Layers<F> {
    Host(Vec<Vec<F>>),
    Device(Box<dyn DeviceTree>, usize),
}

impl<F: Field> Layers<F> {
    /// The power tree over the lifts of `left` by `lift_powers` and the bits of `right`, on
    /// `device` when there is one.
    fn power(
        lift_powers: &[F],
        left: Words<'_>,
        right: Words<'_>,
        rows: usize,
        width: usize,
        depth: usize,
        device: Option<&dyn WordDevice>,
    ) -> Self {
        match device {
            Some(device) => Self::Device(
                device.power_tree(&slice_to_device(lift_powers), left, right, rows, width, depth),
                depth,
            ),
            None => Self::Host(product_layers(
                power_leaves(&lifts(left.host(), lift_powers), right.host(), width),
                depth,
            )),
        }
    }

    /// The bit tree over `columns`, on `device` when there is one.
    fn bits(
        columns: &[Words<'_>],
        powers: &[F],
        rows: usize,
        width: usize,
        depth: usize,
        device: Option<&dyn WordDevice>,
    ) -> Self {
        match device {
            Some(device) => Self::Device(
                device.bit_tree(columns, &slice_to_device(powers), rows, width, depth),
                depth,
            ),
            None => {
                let columns = columns.iter().map(|words| words.host()).collect::<Vec<_>>();
                Self::Host(product_layers(bit_leaves(&columns, powers, width), depth))
            }
        }
    }

    /// The top layer, one product per row, evaluated at `point`.
    fn top_at(&self, point: &[F]) -> F {
        match self {
            Self::Host(layers) => evaluate(&layers[0], point),
            Self::Device(tree, _) => from_device(tree.top_at(&slice_to_device(point))),
        }
    }

    /// The number of layers below the top.
    fn depth(&self) -> usize {
        match self {
            Self::Host(layers) => layers.len() - 1,
            Self::Device(_, depth) => *depth,
        }
    }

    /// Each leaf cell's row extension at `point`.
    fn leaf_values(&self, count: usize, point: &[F]) -> Vec<F> {
        match self {
            Self::Host(layers) => {
                leaf_values(layers.last().expect("a tree has leaves"), count, point)
            }
            Self::Device(tree, _) => tree
                .leaf_values(&slice_to_device(point))
                .into_iter()
                .map(from_device)
                .collect(),
        }
    }
}

/// Runs every layer reduction of one tree, returning the leaf point and claim.
///
/// ```text
///     V_d(r) = sum_z eq(r, z) * V_{d+1}(z, 0) * V_{d+1}(z, 1)
///
///     sumcheck ends at r'     ->  halves h_0 = V_{d+1}(r', 0), h_1 = V_{d+1}(r', 1)
///     line challenge tau      ->  V_{d+1}(r', tau) = h_0 + tau * (h_1 - h_0)
/// ```
#[instrument(skip_all, fields(tree = tree.name()))]
fn prove_layers<C, F, EF>(
    transcript: &mut ProverTranscript<'_, C, F, EF>,
    tree: Tree,
    layers: &mut Layers<EF>,
    mut point: Vec<EF>,
    mut claim: EF,
) -> (Vec<LayerProof<F, EF>>, Vec<EF>, EF)
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let depth = layers.depth();
    let mut proofs = Vec::with_capacity(depth);
    for layer in 0..depth {
        let _span = info_span!("product layer", layer, rounds = point.len()).entered();
        let mut prover = match layers {
            Layers::Host(layers) => {
                // The new variable is the least significant, so halves are even and odd entries.
                let below = &layers[layer + 1];
                let even = below.par_chunks_exact(2).map(|pair| pair[0]).collect();
                let odd = below.par_chunks_exact(2).map(|pair| pair[1]).collect();
                ProductLayer::new(point.clone(), [even, odd], claim)
            }
            Layers::Device(tree, _) => {
                ProductLayer::on_device(point.clone(), &mut **tree, layer, claim)
            }
        };
        let rounds = point.len();
        let ((sumcheck, end), halves, line) = transcript.layer(tree, |challenger| {
            let (proof, end) = prover.prove::<F, _>(challenger, rounds, DEGREE, 0, claim);
            let [zero, one] = prover.terminal();
            ((proof, end), [zero, one])
        });

        claim = halves[0] + line * (halves[1] - halves[0]);
        point = end.as_slice().to_vec();
        point.push(line);
        proofs.push(LayerProof { sumcheck, halves });
    }
    (proofs, point, claim)
}

/// Replays every layer reduction of one tree, returning the leaf point and claim.
fn verify_layers<C, F, EF>(
    transcript: &mut VerifierTranscript<'_, C, F, EF>,
    tree: Tree,
    layers: &[LayerProof<F, EF>],
    mut point: Vec<EF>,
    mut claim: EF,
) -> Result<(Vec<EF>, EF), IntegerMulError>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    for layer in layers {
        // The sum is the claim the previous step left, not one the prover picks.
        if layer.sumcheck.claimed_sum != claim {
            return Err(IntegerMulError::EnteringClaim);
        }
        let rounds = point.len();
        let (replay, line) = transcript.layer(tree, &layer.halves, |challenger| {
            layer.sumcheck.verify(challenger, rounds, DEGREE, 0)
        });
        let (end, value) = replay?;

        // The equality factor is public, so only the two halves are the prover's.
        let [zero, one] = layer.halves;
        if value != Point::eval_eq(&point, end.as_slice()) * zero * one {
            return Err(IntegerMulError::LayerClaim);
        }
        claim = zero + line * (one - zero);
        point = end.as_slice().to_vec();
        point.push(line);
    }
    Ok((point, claim))
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_binary_field::{BinaryChallenger, BinaryField64, BinaryField128};
    use p3_challenger::{CanSample, HashChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use p3_word::{Word, Word32, Word64};
    use proptest::prelude::*;

    use super::*;

    type EF = BinaryField128;
    type Challenger = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;

    fn challenger() -> Challenger {
        Challenger::from_hasher(Vec::new(), Keccak256Hash)
    }

    // Expands one word per row into its bit table, row major and bit minor.
    fn bit_rows(words: &[u64], width: usize) -> Vec<EF> {
        words
            .iter()
            .flat_map(|&word| (0..width).map(move |index| EF::from_bool((word >> index) & 1 == 1)))
            .collect()
    }

    // Packs one row per product into the four operand columns.
    fn columns<W: PackedWord>(rows: &[[W; 4]]) -> [Vec<Packed<W>>; 4] {
        core::array::from_fn(|column| rows.iter().map(|row| row[column].pack()).collect())
    }

    // Proves, verifies, and checks every claim against the column it names.
    fn round_trip<W: PackedWord>(rows: &[[W; 4]]) -> Result<(), IntegerMulError> {
        let reduction = IntegerMulReduction::new(rows.len(), W::BITS.ilog2() as usize).unwrap();
        let columns = columns(rows);
        let mut prover = challenger();
        let (proof, claims) = reduction.prove::<EF, EF, W, _>(Some(&columns), None, &mut prover);
        let mut verifier = challenger();
        let replayed = reduction.verify::<EF, EF, _>(&proof, &mut verifier)?;

        // Both sides end on the same claims and the same sponge state.
        assert_eq!(replayed, claims);
        assert_eq!(
            CanSample::<EF>::sample(&mut prover),
            CanSample::<EF>::sample(&mut verifier)
        );

        // The scalar reference: each claim is its column's bit extension at its point.
        let padded = 1 << reduction.row_variables;
        for (claim, column) in claims.iter().zip(&columns) {
            let words = (0..padded)
                .map(|row| {
                    column
                        .get(row)
                        .map_or(0, |&packed| W::unpack(packed).to_u64())
                })
                .collect::<Vec<_>>();
            let bits = bit_rows(&words, W::BITS as usize);
            assert_eq!(evaluate(&bits, &claim.point), claim.value);
        }
        Ok(())
    }

    fn word64_row(a: u64, b: u64) -> [Word64; 4] {
        let product = u128::from(a) * u128::from(b);
        [a, b, product as u64, (product >> 64) as u64].map(Word64::new)
    }

    fn word32_row(a: u32, b: u32) -> [Word32; 4] {
        let product = u64::from(a) * u64::from(b);
        [a, b, product as u32, (product >> 32) as u32].map(Word32::new)
    }

    #[test]
    fn the_row_cube_is_padded_to_a_power_of_two() {
        // No product means no reduction at all.
        assert_eq!(IntegerMulReduction::new(0, 6), None);

        // One product needs no row variable, three need two, and four still need two.
        assert_eq!(IntegerMulReduction::new(1, 6).unwrap().row_variables, 0);
        assert_eq!(IntegerMulReduction::new(3, 6).unwrap().row_variables, 2);
        assert_eq!(IntegerMulReduction::new(4, 5).unwrap().row_variables, 2);
    }

    #[test]
    fn a_field_too_small_for_the_product_width_is_refused() {
        // 64-bit words need a field of order at least 2^128.
        let word64 = IntegerMulReduction::new(1, 6).unwrap();
        assert_eq!(word64.check_field::<EF>(), Ok(()));
        assert_eq!(
            word64.check_field::<BinaryField64>(),
            Err(IntegerMulError::FieldTooSmall { required: 128 })
        );

        // 32-bit words fit a field of order 2^64 exactly.
        let word32 = IntegerMulReduction::new(1, 5).unwrap();
        assert_eq!(word32.check_field::<BinaryField64>(), Ok(()));
    }

    #[test]
    fn the_verifier_refuses_a_field_too_small_for_the_product_width() {
        // Fixture state: one 64-bit product, 2^33 * 2^32 = 2^65, claimed as lo = 2, hi = 0.
        //
        //     2^65 = 2 * 2^64 = 2  (mod 2^64 - 1)
        //
        // Over GF(2^64) both lifts are g^2, and the low bit agrees, so only the field check refuses.
        type Small = BinaryField64;
        let small = || BinaryChallenger::<Small, _>::from_hasher(Vec::new(), Keccak256Hash);
        let row = [1 << 33, 1 << 32, 2, 0].map(Word64::new);
        let reduction = IntegerMulReduction::new(1, 6).unwrap();

        // The reduction's prover does not check the field, so it produces a record.
        let (proof, _) =
            reduction.prove::<Small, Small, Word64, _>(Some(&columns(&[row])), None, &mut small());

        // The verifier refuses it on its own.
        assert_eq!(
            reduction.verify::<Small, Small, _>(&proof, &mut small()),
            Err(IntegerMulError::FieldTooSmall { required: 128 })
        );
    }

    #[test]
    fn a_power_tree_over_other_factors_is_refused() {
        // Fixture state: 3 * 5 claimed as 17, and 7 * 9 claimed as 63.
        //
        // Both claimed products are odd, so the low bit agrees.
        let rows = [[3, 5, 17, 0], [7, 9, 63, 0]].map(|row| row.map(Word64::new));
        let reduction = IntegerMulReduction::new(2, 6).unwrap();
        let words = columns(&rows).map(|column| {
            column
                .iter()
                .map(|&packed| Word64::unpack(packed).to_u64())
                .collect::<Vec<_>>()
        });
        let ones = vec![1; 2];

        // The forger walks the power tree over (17, 1) and (63, 1), so both lifts agree.
        //
        // The selector then runs over the true factors.
        let mut sponge = challenger();
        let mut transcript =
            ProverTranscript::<Challenger, EF, EF>::new(&mut sponge, reduction.transcript_shape());
        let (proof, _) = reduction.prove_words(
            &mut transcript,
            words.each_ref().map(|words| Words::Host(words)),
            [Words::Host(&words[2]), Words::Host(&ones)],
            None,
        );
        transcript.finish();

        // The selector's sum follows the forged leaves, so only its closing check sees the factors.
        assert_eq!(
            reduction.verify::<EF, EF, _>(&proof, &mut challenger()),
            Err(IntegerMulError::SelectorClaim)
        );
    }

    #[test]
    fn the_twist_inverts_the_frobenius_power() {
        // Invariant: phi^(-i)(y^(2^i)) = y for every i below the degree.
        let degree = binary_degree::<EF>().unwrap();
        assert_eq!(degree, 128);
        let y = EF::GENERATOR * EF::GENERATOR + EF::ONE;
        for i in [0, 1, 5, 63, 127] {
            assert_eq!(twist(y.exp_power_of_2(i), i, degree), y);
        }
    }

    #[test]
    fn a_single_product_needs_no_row_point() {
        // Fixture state: the largest squares, whose limbs are both nearly all ones.
        round_trip(&[word64_row(u64::MAX, u64::MAX)]).unwrap();
        round_trip(&[word32_row(u32::MAX, u32::MAX)]).unwrap();
    }

    #[test]
    fn a_zero_product_lifts_to_one_on_both_sides() {
        // Every leaf of a zero factor is one, as is every leaf of a zero result.
        round_trip(&[word64_row(0, u64::MAX), word64_row(u64::MAX, 0)]).unwrap();
    }

    #[test]
    fn a_false_product_is_rejected() {
        // Mutation: the high limb of the second row is off by one.
        let mut rows = [word32_row(7, 9), word32_row(u32::MAX, 3), word32_row(5, 5)];
        rows[1][3] = Word32::new(rows[1][3].get() + 1);
        assert!(round_trip(&rows).is_err());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]

        #[test]
        fn random_word32_products_round_trip(
            rows in prop::collection::vec(any::<(u32, u32)>(), 1..6),
        ) {
            let rows = rows.into_iter().map(|(a, b)| word32_row(a, b)).collect::<Vec<_>>();
            prop_assert_eq!(round_trip(&rows), Ok(()));
        }

        #[test]
        fn random_word64_products_round_trip(
            rows in prop::collection::vec(any::<(u64, u64)>(), 1..4),
        ) {
            let rows = rows.into_iter().map(|(a, b)| word64_row(a, b)).collect::<Vec<_>>();
            prop_assert_eq!(round_trip(&rows), Ok(()));
        }
    }
}
