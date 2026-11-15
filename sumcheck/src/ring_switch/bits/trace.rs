//! Bit unpacking through the field trace, with no sumcheck of its own.
//!
//! # Overview
//!
//! A packed element `X_i` holds `m` witness bits `x_{i,j}` as coordinates in the basis `beta`:
//!
//! ```text
//!     X_i = sum_j beta_j * x_{i,j}
//! ```
//!
//! The trace `Tr : F -> F_2` is `F_2`-linear, and the dual basis `alpha` of `beta` satisfies
//! `Tr(alpha_k * beta_j) = [k = j]`.  So `Tr(alpha_j * X_i) = x_{i,j}`: the trace plucks a bit.
//!
//! A field claim `Y = sum_{i,j} R_{i,j} * x_{i,j}` with coefficients `R_{i,j}` in the challenge
//! field `EF` splits into one bit claim per coordinate `k` of `EF`, and each bit claim is a
//! trace-zero condition on a linear form of the packed vector:
//!
//! ```text
//!     Z_k = sum_i Q_{i,k} * X_i + rho * d_k(Y)         Q_{i,k} = sum_j d_k(R_{i,j}) * alpha_j
//!     Tr(Z_k) = 0  for every k                         rho an element of trace one
//! ```
//!
//! The prover sends every `Z_k`, the verifier checks the traces and draws `gamma`, and what is
//! left is one linear relation over the packed vector:
//!
//! ```text
//!     sum_k gamma^k * (Z_k + rho * d_k(Y))  =  sum_i r'_i * X_i      r'_i = sum_k gamma^k * Q_{i,k}
//! ```
//!
//! A commitment answering linear relations discharges it directly.  WHIR answers it as a weighted
//! claim, so this module hands out the weight `r'` two ways: as a table over the packed cube for
//! the prover, and as an evaluator of its multilinear extension for the verifier.
//!
//! This is the trace-based packing protocol of Frigo and shelat (ePrint 2026/2770), Figure 1.
//! Against the ring switch in [`super::reduction`] it sends one message for every claim of an
//! opening rather than one element per claim, runs no rounds, and needs no tensor structure in
//! the coefficients, which is what lets the claims be batched before anything is sent.
//!
//! # The claims handled
//!
//! Every claim is a [`BitRingSwitch`] set up at its own point: `R_{i,j} = C_i * D_j` with `C` the
//! equality table of the kept coordinates and `D` that of the absorbed ones.  With that structure
//! `Q_{i,k} = sum_l d_l(C_i) * mu_{k,l}`, where `mu` depends on the point and the batching weight
//! alone, and `sum_i Q_{i,k} X_i = sum_l mu_{k,l} * W_l` with `W_l = sum_i d_l(C_i) X_i` the rows
//! of the tensor element the ring switch already accumulates.  So the prover's work is the same
//! tensors and a `dim EF x dim EF` matrix per claim.
//!
//! Successor readings are not handled here yet; a claim asking for one is refused.
//!
//! # The verifier's weight
//!
//! Over Boolean `i` the bit `d_l(C_i)` is `Tr(alpha_l * C_i) = sum_s (alpha_l * C_i)^(2^s)`, and
//! `eq(r, i)^(2^s) = eq(Frob^s(r), i)` in characteristic two, so the multilinear extension of
//! `i -> r'_i` is a sum over the Frobenius orbit of the kept coordinates:
//!
//! ```text
//!     w(z) = sum_claims sum_s Gamma_s * A_s * eq(Frob^s(r), z)
//!
//!     Gamma_s = sum_k gamma^k * alpha_k^(2^s)          once per batch
//!     A_s     = sum_j alpha_j * (lambda * D_j)^(2^s)    once per claim
//! ```
//!
//! The split follows from `sum_l beta_l * alpha_l^(2^t) = [t = 0]`, which holds for any pair
//! of dual bases.  Both factors are read off with no products beyond the `dim F` that form
//! `lambda * D_j`: multiplying by the constant `alpha_j` and squaring are `F_2`-linear maps of
//! `EF`, so the verifier applies them as byte-table lookups, `dim F * dim EF` of each per
//! claim, and `Gamma_s` costs `dim EF` products per Frobenius power once for the whole batch.
//! Evaluating the weight costs `dim EF` equality evaluations per claim, and the Frobenius
//! conjugates of a Boolean prefix are itself, so a slot address needs nothing special.
//!
//! # Soundness
//!
//! Binding every value before `lambda` is drawn, and every `Z_k` before `gamma`, the reduction's
//! own error is `(L - 1) / |EF|` for the claim batching and `(dim EF - 1) / |EF|` for the powers
//! of `gamma` (Lemma 2.3 of the paper with powers in place of independent draws), per attempt.
//! A wrong value leaves some `Z_k` with trace one, which is either sent as it is and rejected,
//! or shifted and caught by the batched linear relation.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{BitCoordinates, TowerLevel};
use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use serde::{Deserialize, Serialize};

use super::basis::{Coefficients, CoordinateSums};
use super::packing::BitPacking;
use super::reduction::BitRingSwitch;
use super::tensor::BitTensor;
use crate::layout::InitialWeight;
use crate::product_polynomial::ProductPolynomial;
use crate::strategy::{SumcheckProver, VariableOrder};

/// A row of up to 192 bits, for linear algebra over `F_2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BitRow([u64; 3]);

impl BitRow {
    const fn get(&self, index: usize) -> bool {
        (self.0[index / 64] >> (index % 64)) & 1 == 1
    }

    const fn set(&mut self, index: usize) {
        self.0[index / 64] |= 1 << (index % 64);
    }

    fn xor_assign(&mut self, other: &Self) {
        for (a, b) in self.0.iter_mut().zip(other.0) {
            *a ^= b;
        }
    }
}

/// The trace form of a binary field in the basis its coordinates define.
///
/// Holds the trace of every basis element, the dual basis, and one element of trace one.
#[derive(Clone, Debug)]
pub struct TraceBasis<K> {
    /// `Tr(beta_j)` for every coordinate `j`.
    mask: BitRow,
    /// `alpha_k` with `Tr(alpha_k * beta_j) = [k = j]`.
    dual: Vec<K>,
    /// `alpha_k^(2^s)` for every `k` and `s`, row `k`.
    dual_conjugates: Vec<Vec<K>>,
    /// An element of trace one.
    rho: K,
}

impl<K: BitCoordinates> TraceBasis<K> {
    /// The trace form of `K`.
    ///
    /// # Panics
    ///
    /// Panics if the trace form is degenerate, which no finite field's is.
    #[must_use]
    pub fn new() -> Self {
        let dimension = Coefficients::<K>::DIMENSION;
        assert!(
            dimension <= 192,
            "the bit rows hold at most 192 coordinates"
        );
        let basis: Vec<K> = (0..dimension).map(Self::basis_element).collect();
        let traces: Vec<bool> = basis.iter().map(|&b| Self::absolute_trace(b)).collect();
        let mut mask = BitRow::default();
        for (j, &t) in traces.iter().enumerate() {
            if t {
                mask.set(j);
            }
        }
        // The Gram matrix of the trace form, inverted over F_2 by elimination on [G | I].
        let mut gram: Vec<BitRow> = basis
            .iter()
            .map(|&bi| {
                let mut row = BitRow::default();
                for (j, &bj) in basis.iter().enumerate() {
                    if Self::absolute_trace(bi * bj) {
                        row.set(j);
                    }
                }
                row
            })
            .collect();
        let mut inverse: Vec<BitRow> = (0..dimension)
            .map(|i| {
                let mut row = BitRow::default();
                row.set(i);
                row
            })
            .collect();
        for column in 0..dimension {
            let pivot = (column..dimension)
                .find(|&row| gram[row].get(column))
                .expect("the trace form of a finite field is non-degenerate");
            gram.swap(column, pivot);
            inverse.swap(column, pivot);
            let (pivot_gram, pivot_inverse) = (gram[column], inverse[column]);
            for row in 0..dimension {
                if row != column && gram[row].get(column) {
                    gram[row].xor_assign(&pivot_gram);
                    inverse[row].xor_assign(&pivot_inverse);
                }
            }
        }
        let dual: Vec<K> = inverse
            .iter()
            .map(|row| {
                let mut coefficients = Coefficients::<K>::zero();
                for i in 0..dimension {
                    if row.get(i) {
                        coefficients.set(i);
                    }
                }
                coefficients.element()
            })
            .collect();
        let dual_conjugates = dual
            .iter()
            .map(|&alpha| {
                let mut conjugates = Vec::with_capacity(dimension);
                let mut power = alpha;
                for _ in 0..dimension {
                    conjugates.push(power);
                    power = power.square();
                }
                conjugates
            })
            .collect();
        let rho = basis[traces
            .iter()
            .position(|&t| t)
            .expect("some basis element has trace one")];
        Self {
            mask,
            dual,
            dual_conjugates,
            rho,
        }
    }

    /// The element whose only set coordinate is `index`.
    fn basis_element(index: usize) -> K {
        let mut coefficients = Coefficients::<K>::zero();
        coefficients.set(index);
        coefficients.element()
    }

    /// `Tr(x)` by its definition as the sum of the Frobenius conjugates.
    fn absolute_trace(x: K) -> bool {
        let mut acc = K::ZERO;
        let mut power = x;
        for _ in 0..Coefficients::<K>::DIMENSION {
            acc += power;
            power = power.square();
        }
        assert!(
            acc == K::ZERO || acc == K::ONE,
            "the trace of a binary field element is a bit"
        );
        acc == K::ONE
    }

    /// `Tr(x)`, read as the parity of the coordinates of `x` the mask selects.
    #[must_use]
    pub fn trace(&self, x: K) -> bool {
        Coefficients::<K>::of(x)
            .iter_set()
            .filter(|&j| self.mask.get(j))
            .count()
            % 2
            == 1
    }

    /// The dual basis: `Tr(alpha_k * beta_j) = [k = j]`.
    #[must_use]
    pub fn dual(&self) -> &[K] {
        &self.dual
    }

    /// `alpha_k^(2^s)`.
    #[must_use]
    pub fn dual_conjugate(&self, k: usize, s: usize) -> K {
        self.dual_conjugates[k][s]
    }

    /// An element of trace one.
    #[must_use]
    pub const fn rho(&self) -> K {
        self.rho
    }
}

impl<K: BitCoordinates> Default for TraceBasis<K> {
    fn default() -> Self {
        Self::new()
    }
}

/// What can go wrong setting up or verifying a trace reduction.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TraceError {
    /// No claims were given.
    #[error("a trace reduction needs at least one claim")]
    NoClaims,
    /// The claims are not all over the same packed cube.
    #[error("claim {claim} leaves {actual} packed variables, the first claim {expected}")]
    Arity {
        /// Index of the offending claim.
        claim: usize,
        /// Packed variables of the first claim.
        expected: usize,
        /// Packed variables of this claim.
        actual: usize,
    },
    /// A claim asks for a successor reading, which this reduction does not answer yet.
    #[error("claim {claim} asks for a successor reading")]
    Successor {
        /// Index of the offending claim.
        claim: usize,
    },
    /// The number of values or tensors does not match the number of claims.
    #[error("{actual} items against {expected} claims")]
    Count {
        /// Number of claims.
        expected: usize,
        /// Number of items supplied.
        actual: usize,
    },
    /// The proof does not carry one trace message per coordinate of the challenge field.
    #[error("{actual} trace messages, {expected} expected")]
    Messages {
        /// Coordinates of the challenge field.
        expected: usize,
        /// Messages in the proof.
        actual: usize,
    },
    /// A trace message has trace one, so some claimed value is wrong.
    #[error("trace message {index} has trace one")]
    TraceOne {
        /// Index of the offending message.
        index: usize,
    },
}

/// The prover's message: `Z_k` for every coordinate `k` of the challenge field.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct TraceProof<F> {
    /// `Z_k`, in coordinate order.
    pub traces: Vec<F>,
}

/// One claim's share of the surviving weight.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ClaimWeight<EF> {
    /// The kept coordinates of the point, the Boolean prefix included.
    high: Vec<EF>,
    /// Where this claim's run begins in the packed cube.
    offset: usize,
    /// Coordinates the prefix fixes.
    prefix: usize,
    /// `Gamma_s * A_s`, one per Frobenius power `s`.
    frobenius: Vec<EF>,
}

/// The claim that survives: `sum_i r'_i X_i = sum`, with `r'` given two ways.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceWeights<EF> {
    claims: Vec<ClaimWeight<EF>>,
    sum: EF,
    num_variables: usize,
}

impl<EF: BitCoordinates> TraceWeights<EF> {
    /// The value the weighted sum over the packed vector must take.
    #[must_use]
    pub const fn sum(&self) -> EF {
        self.sum
    }

    /// Variables of the packed cube the weight is over.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// The multilinear extension of `i -> r'_i`, read at `z`.
    ///
    /// # Panics
    ///
    /// Panics unless `z` has the packed cube's variables.
    #[must_use]
    pub fn weight_at(&self, z: &Point<EF>) -> EF {
        assert_eq!(
            z.num_variables(),
            self.num_variables,
            "the weight is read over the packed cube"
        );
        let mut conjugate = vec![EF::ZERO; self.num_variables];
        self.claims
            .iter()
            .map(|claim| {
                conjugate.copy_from_slice(&claim.high);
                let mut total = EF::ZERO;
                for &g_s in &claim.frobenius {
                    if g_s != EF::ZERO {
                        total += g_s * Point::<EF>::eval_eq(&conjugate, z.as_slice());
                    }
                    for coordinate in conjugate.iter_mut() {
                        *coordinate = coordinate.square();
                    }
                }
                total
            })
            .sum()
    }

    /// `r'_i` for every `i` of the packed cube, which the prover's weighted sumcheck starts from.
    ///
    /// Within one claim `r'_i = L(C_i)` for the `F_2`-linear map `L(x) = sum_s G_s x^(2^s)`, so the
    /// table is `L` read on the basis, `g_l = L(beta_l)`, applied as byte tables to the equality
    /// table of the kept coordinates.
    pub fn weight_table(&self) -> Poly<EF> {
        let mut table = Poly::<EF>::zero(self.num_variables);
        for claim in &self.claims {
            let g: Vec<EF> = (0..Coefficients::<EF>::DIMENSION)
                .map(|l| {
                    let mut power = TraceBasis::<EF>::basis_element(l);
                    let mut g_l = EF::ZERO;
                    for &g_s in &claim.frobenius {
                        g_l += g_s * power;
                        power = power.square();
                    }
                    g_l
                })
                .collect();
            let sums = CoordinateSums::<EF, EF>::new(&g);
            let equality = Poly::<EF>::new_from_point(&claim.high[claim.prefix..], EF::ONE);
            for (slot, &c) in table.as_mut_slice()
                [claim.offset..claim.offset + equality.num_evals()]
                .iter_mut()
                .zip(equality.as_slice())
            {
                *slot += sums.sum(c);
            }
        }
        table
    }

    /// The relation as the initial weight of a proximity opening of `packing`.
    ///
    /// The opening's first fold proves `sum_i r'_i X_i = sum` beside the commitment's own
    /// claims, so no point survives this reduction and nothing of it is bound separately.
    ///
    /// # Panics
    ///
    /// Panics unless the packing has the weight's variables.
    pub fn initial_weight<F, S>(&self, packing: &BitPacking<F, S>) -> TraceWeight<EF>
    where
        F: TowerLevel + BitCoordinates,
        EF: ExtensionField<F>,
        S: core::borrow::Borrow<[F]>,
    {
        assert_eq!(
            packing.num_variables(),
            self.num_variables,
            "the weight is over the packed cube"
        );
        let evals = Poly::new(
            packing
                .poly()
                .as_slice()
                .iter()
                .map(|&x| EF::from(x))
                .collect(),
        );
        TraceWeight {
            prover: SumcheckProver::new(
                ProductPolynomial::<EF, EF>::new_unpacked(
                    VariableOrder::Suffix,
                    evals,
                    self.weight_table(),
                ),
                self.sum,
            ),
            message: (EF::ZERO, EF::ZERO),
            bound: 0,
            sum: self.sum,
        }
    }
}

/// The surviving relation measured against the packed polynomial, round by round.
///
/// A proximity opening's initial fold binds the trailing variables first, so the product
/// is held in suffix order; what is left after the fold is the weight over the leading ones.
#[derive(Debug)]
pub struct TraceWeight<EF: Field> {
    /// The packed polynomial against the weight table.
    prover: SumcheckProver<EF, EF>,
    /// The last message measured, which the next challenge reduces the claim by.
    message: (EF, EF),
    /// Challenges applied so far.
    bound: usize,
    /// `sum_i r'_i X_i`.
    sum: EF,
}

impl<EF: Field> TraceWeight<EF> {
    /// Apply the challenges not applied yet, at most the one the last measure left.
    fn catch_up(&mut self, challenges: &[EF]) {
        debug_assert!(
            challenges.len() <= self.bound + 1,
            "one challenge per round"
        );
        if let Some(&r) = challenges.get(self.bound) {
            let (c0, c_inf) = self.message;
            self.prover.reduce_claim_with_coefficients(c0, c_inf, r);
            self.prover.hold(r);
            self.bound += 1;
        }
    }
}

impl<EF: Field> InitialWeight<EF> for TraceWeight<EF> {
    fn claimed_sum(&self) -> EF {
        self.sum
    }

    fn round(&mut self, challenges: &[EF]) -> (EF, EF) {
        self.catch_up(challenges);
        self.message = self.prover.measure_round();
        self.message
    }

    fn into_folded(mut self, challenges: &[EF]) -> Poly<EF> {
        self.catch_up(challenges);
        self.prover.weights()
    }
}

/// Several claims about one packing, reduced together through the trace.
#[derive(Clone, Debug)]
pub struct TraceReduction<F, EF = F> {
    reductions: Vec<BitRingSwitch<F, EF>>,
    packed: TraceBasis<F>,
    challenge: TraceBasis<EF>,
    /// `x -> alpha_j * x` on `EF` for every `j`, as byte tables.
    alpha_multipliers: Vec<CoordinateSums<EF, EF>>,
    /// `x -> x^2` on `EF`, as byte tables.
    squaring: CoordinateSums<EF, EF>,
}

impl<F, EF> TraceReduction<F, EF>
where
    F: TowerLevel + BitCoordinates,
    EF: BitCoordinates + ExtensionField<F>,
{
    /// Set the reduction up over `reductions`, one per claim, each at its own point.
    ///
    /// # Errors
    ///
    /// Returns an error without claims, when they are over different packed cubes, or when one
    /// asks for a successor reading.
    pub fn new(reductions: Vec<BitRingSwitch<F, EF>>) -> Result<Self, TraceError> {
        let Some(first) = reductions.first() else {
            return Err(TraceError::NoClaims);
        };
        let expected = first.num_variables();
        for (claim, reduction) in reductions.iter().enumerate() {
            if reduction.num_variables() != expected {
                return Err(TraceError::Arity {
                    claim,
                    expected,
                    actual: reduction.num_variables(),
                });
            }
            if reduction.successor().is_some() {
                return Err(TraceError::Successor { claim });
            }
        }
        let packed = TraceBasis::<F>::new();
        let basis: Vec<EF> = (0..Coefficients::<EF>::DIMENSION)
            .map(TraceBasis::<EF>::basis_element)
            .collect();
        let alpha_multipliers = packed
            .dual()
            .iter()
            .map(|&alpha| {
                let scaled: Vec<EF> = basis.iter().map(|&b| b * alpha).collect();
                CoordinateSums::new(&scaled)
            })
            .collect();
        let squares: Vec<EF> = basis.iter().map(|b| b.square()).collect();
        Ok(Self {
            reductions,
            packed,
            challenge: TraceBasis::new(),
            alpha_multipliers,
            squaring: CoordinateSums::new(&squares),
        })
    }

    /// The claims, in the order they are bound.
    #[must_use]
    pub fn reductions(&self) -> &[BitRingSwitch<F, EF>] {
        &self.reductions
    }

    /// Variables of the packed cube.
    #[must_use]
    pub fn num_variables(&self) -> usize {
        self.reductions[0].num_variables()
    }

    /// The tensor element of every claim, as the ring switch accumulates it.
    ///
    /// # Panics
    ///
    /// Panics unless the packing has the claims' variables.
    #[must_use]
    pub fn tensors<S: core::borrow::Borrow<[F]>>(
        &self,
        packing: &BitPacking<F, S>,
    ) -> Vec<BitTensor<EF, F>>
    where
        F: Send + Sync,
        EF: Send + Sync,
    {
        assert_eq!(
            packing.num_variables(),
            self.num_variables(),
            "the packing must have the variables the evaluation points leave"
        );
        self.reductions
            .iter()
            .map(|reduction| {
                let (_, offset, equality) = reduction.support();
                BitRingSwitch::tensor_over(packing, offset, &equality)
            })
            .collect()
    }

    /// The value each claim takes, read off its tensor element.
    #[must_use]
    pub fn values(&self, tensors: &[BitTensor<EF, F>]) -> Vec<EF> {
        self.reductions
            .iter()
            .zip(tensors)
            .map(|(reduction, tensor)| reduction.incoming_claim(tensor))
            .collect()
    }

    /// The equality table of a claim's absorbed coordinates: `D_j`.
    fn low_weights(reduction: &BitRingSwitch<F, EF>) -> Poly<EF> {
        let point = reduction.point();
        let (_, low) = point.split_at(point.num_variables() - BitRingSwitch::<F, EF>::ABSORBED);
        Poly::new_from_point(low.as_slice(), EF::ONE)
    }

    /// `mu_{k,l} = sum_j d_k(lambda * beta_l * D_j) * alpha_j`, row `k`, for one claim.
    ///
    /// The prover forms its trace messages from these; the verifier never needs them.
    fn mu(&self, reduction: &BitRingSwitch<F, EF>, lambda: EF) -> Vec<Vec<F>> {
        let low = Self::low_weights(reduction);
        let mut mu =
            vec![vec![F::ZERO; Coefficients::<EF>::DIMENSION]; Coefficients::<EF>::DIMENSION];
        for (l, scaled) in (0..Coefficients::<EF>::DIMENSION)
            .map(|l| (l, lambda * TraceBasis::<EF>::basis_element(l)))
        {
            for (&d_j, &alpha_j) in low.as_slice().iter().zip(self.packed.dual()) {
                for k in Coefficients::<EF>::of(scaled * d_j).iter_set() {
                    mu[k][l] += alpha_j;
                }
            }
        }
        mu
    }

    /// Bind the points and values and draw the claim batching challenge.
    fn bind_values<Challenger: FieldChallenger<F>>(
        &self,
        values: &[EF],
        challenger: &mut Challenger,
    ) -> Result<EF, TraceError> {
        if values.len() != self.reductions.len() {
            return Err(TraceError::Count {
                expected: self.reductions.len(),
                actual: values.len(),
            });
        }
        for (reduction, &value) in self.reductions.iter().zip(values) {
            challenger.observe_algebra_slice(reduction.point().as_slice());
            challenger.observe_algebra_element(value);
        }
        Ok(challenger.sample_algebra_element())
    }

    /// The powers of a challenge, one per item.
    fn powers(base: EF, count: usize) -> Vec<EF> {
        let mut powers = Vec::with_capacity(count);
        let mut power = EF::ONE;
        for _ in 0..count {
            powers.push(power);
            power *= base;
        }
        powers
    }

    /// The surviving weighted claim both sides compute once `gamma` is drawn.
    ///
    /// `Gamma_s` takes `dim EF` products per Frobenius power for the whole batch; a claim then
    /// takes `dim F` products for `lambda * D_j` and byte-table lookups for everything else.
    fn weights(&self, lambdas: &[EF], gamma: EF, sum: EF) -> TraceWeights<EF> {
        let gammas = Self::powers(gamma, Coefficients::<EF>::DIMENSION);
        let gamma_conjugates: Vec<EF> = (0..Coefficients::<EF>::DIMENSION)
            .map(|s| {
                gammas
                    .iter()
                    .enumerate()
                    .map(|(k, &gamma_k)| gamma_k * self.challenge.dual_conjugate(k, s))
                    .sum()
            })
            .collect();
        let claims = self
            .reductions
            .iter()
            .zip(lambdas)
            .map(|(reduction, &lambda)| {
                let low = Self::low_weights(reduction);
                let mut conjugates: Vec<EF> = low.as_slice().iter().map(|&d| lambda * d).collect();
                let frobenius = gamma_conjugates
                    .iter()
                    .map(|&gamma_s| {
                        let a_s: EF = conjugates
                            .iter()
                            .zip(&self.alpha_multipliers)
                            .map(|(&u, multiplier)| multiplier.sum(u))
                            .sum();
                        for u in conjugates.iter_mut() {
                            *u = self.squaring.sum(*u);
                        }
                        gamma_s * a_s
                    })
                    .collect();
                let (prefix, address) = reduction.fixed_prefix();
                let offset = address << (self.num_variables() - prefix);
                ClaimWeight {
                    high: reduction.high().to_vec(),
                    offset,
                    prefix,
                    frobenius,
                }
            })
            .collect();
        TraceWeights {
            claims,
            sum,
            num_variables: self.num_variables(),
        }
    }

    /// `sum_k gamma^k (Z_k + rho * d_k(Y))`.
    fn batched_sum(&self, traces: &[F], batched_value: EF, gamma: EF) -> EF {
        let bits = Coefficients::<EF>::of(batched_value);
        let rho = self.packed.rho();
        Self::powers(gamma, Coefficients::<EF>::DIMENSION)
            .iter()
            .zip(traces)
            .enumerate()
            .map(|(k, (&gamma_k, &z_k))| {
                let shifted = if bits.get(k) { z_k + rho } else { z_k };
                gamma_k * EF::from(shifted)
            })
            .sum()
    }

    /// Reduce the claims whose tensors are `tensors` and whose values are `values`.
    ///
    /// # Returns
    ///
    /// The proof and the surviving weighted claim.
    ///
    /// # Errors
    ///
    /// Returns an error unless there is one tensor and one value per claim.
    pub fn prove<Challenger: FieldChallenger<F>>(
        &self,
        tensors: &[BitTensor<EF, F>],
        values: &[EF],
        challenger: &mut Challenger,
    ) -> Result<(TraceProof<F>, TraceWeights<EF>), TraceError> {
        if tensors.len() != self.reductions.len() {
            return Err(TraceError::Count {
                expected: self.reductions.len(),
                actual: tensors.len(),
            });
        }
        let lambda = self.bind_values(values, challenger)?;
        let lambdas = Self::powers(lambda, self.reductions.len());
        let batched_value: EF = lambdas.iter().zip(values).map(|(&l, &v)| l * v).sum();
        let bits = Coefficients::<EF>::of(batched_value);
        let rho = self.packed.rho();
        let mus: Vec<Vec<Vec<F>>> = self
            .reductions
            .iter()
            .zip(&lambdas)
            .map(|(reduction, &lambda_l)| self.mu(reduction, lambda_l))
            .collect();
        let traces: Vec<F> = (0..Coefficients::<EF>::DIMENSION)
            .map(|k| {
                let mut z_k = if bits.get(k) { rho } else { F::ZERO };
                for (mu, tensor) in mus.iter().zip(tensors) {
                    for (&mu_kl, &w_l) in mu[k].iter().zip(tensor.rows()) {
                        z_k += mu_kl * w_l;
                    }
                }
                z_k
            })
            .collect();
        challenger.observe_algebra_slice(&traces);
        let gamma: EF = challenger.sample_algebra_element();
        let sum = self.batched_sum(&traces, batched_value, gamma);
        Ok((TraceProof { traces }, self.weights(&lambdas, gamma, sum)))
    }

    /// Check the proof of claims with values `values`, leaving the surviving weighted claim.
    ///
    /// # Errors
    ///
    /// Returns an error unless there is one value per claim, the proof has one message per
    /// coordinate of the challenge field, and every message has trace zero.
    pub fn verify<Challenger: FieldChallenger<F>>(
        &self,
        values: &[EF],
        proof: &TraceProof<F>,
        challenger: &mut Challenger,
    ) -> Result<TraceWeights<EF>, TraceError> {
        if proof.traces.len() != Coefficients::<EF>::DIMENSION {
            return Err(TraceError::Messages {
                expected: Coefficients::<EF>::DIMENSION,
                actual: proof.traces.len(),
            });
        }
        let lambda = self.bind_values(values, challenger)?;
        let lambdas = Self::powers(lambda, self.reductions.len());
        let batched_value: EF = lambdas.iter().zip(values).map(|(&l, &v)| l * v).sum();
        challenger.observe_algebra_slice(&proof.traces);
        if let Some(index) = proof.traces.iter().position(|&z| self.packed.trace(z)) {
            return Err(TraceError::TraceOne { index });
        }
        let gamma: EF = challenger.sample_algebra_element();
        let sum = self.batched_sum(&proof.traces, batched_value, gamma);
        Ok(self.weights(&lambdas, gamma, sum))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::vec::Vec;

    use p3_binary_field::{BinaryChallenger, BinaryField8, BinaryField16, BinaryField128};
    use p3_challenger::HashChallenger;
    use p3_field::FieldAlgebra;
    use p3_keccak::Keccak256Hash;
    use rand::distr::{Distribution, StandardUniform};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    fn dual_basis_check<K: BitCoordinates>()
    where
        StandardUniform: Distribution<K>,
    {
        let basis = TraceBasis::<K>::new();
        assert!(basis.trace(basis.rho()));
        for k in 0..Coefficients::<K>::DIMENSION {
            for j in 0..Coefficients::<K>::DIMENSION {
                let beta_j = TraceBasis::<K>::basis_element(j);
                assert_eq!(basis.trace(basis.dual()[k] * beta_j), k == j, "k {k} j {j}");
            }
        }
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..16 {
            let x: K = rng.random();
            let bits = Coefficients::<K>::of(x);
            for k in 0..Coefficients::<K>::DIMENSION {
                assert_eq!(basis.trace(basis.dual()[k] * x), bits.get(k));
            }
            assert_eq!(basis.trace(x), TraceBasis::<K>::absolute_trace(x));
        }
    }

    #[test]
    fn dual_basis_is_dual_and_plucks_bits() {
        dual_basis_check::<BinaryField8>();
        dual_basis_check::<BinaryField16>();
        dual_basis_check::<BinaryField128>();
    }

    type Chal<F> = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

    fn challenger<F: TowerLevel>() -> Chal<F> {
        Chal::<F>::from_hasher(Vec::new(), Keccak256Hash)
    }

    /// The bit witness, the claims, and their points: two plain points and one with a prefix.
    fn fixture<F, EF>(packed_variables: usize, seed: u64) -> (BitPacking<F>, TraceReduction<F, EF>)
    where
        F: TowerLevel + BitCoordinates,
        EF: BitCoordinates + ExtensionField<F>,
        StandardUniform: Distribution<EF>,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let num_variables = packed_variables + BitRingSwitch::<F, EF>::ABSORBED;
        let bits: Vec<u8> = (0..1usize << (num_variables - 3))
            .map(|_| rng.random::<u8>())
            .collect();
        let packing = BitPacking::<F>::new(&bits).unwrap();
        let mut points = vec![
            Point::<EF>::rand(&mut rng, num_variables),
            Point::<EF>::rand(&mut rng, num_variables),
        ];
        let mut prefixed: Vec<EF> = Point::<EF>::rand(&mut rng, num_variables)
            .as_slice()
            .to_vec();
        prefixed[0] = EF::ONE;
        prefixed[1] = EF::ZERO;
        points.push(Point::new(prefixed));
        let reductions = points
            .iter()
            .map(|point| BitRingSwitch::<F, EF>::new(point).unwrap())
            .collect();
        (packing, TraceReduction::new(reductions).unwrap())
    }

    /// The bit witness read as a multilinear over every variable, at `point`.
    fn bit_evaluation<F, EF>(packing: &BitPacking<F>, point: &Point<EF>) -> EF
    where
        F: TowerLevel + BitCoordinates,
        EF: BitCoordinates + ExtensionField<F>,
    {
        let absorbed = BitRingSwitch::<F, EF>::ABSORBED;
        let equality = Poly::<EF>::new_from_point(point.as_slice(), EF::ONE);
        let mut total = EF::ZERO;
        for i in 0..packing.len() {
            let coefficients = packing.coefficients(i);
            for j in coefficients.iter_set() {
                total += equality.as_slice()[(i << absorbed) | j];
            }
        }
        total
    }

    fn round_trip<F, EF>(packed_variables: usize, seed: u64)
    where
        F: TowerLevel + BitCoordinates + Send + Sync,
        EF: BitCoordinates + ExtensionField<F> + Send + Sync,
        StandardUniform: Distribution<EF>,
    {
        let (packing, reduction) = fixture::<F, EF>(packed_variables, seed);
        let tensors = reduction.tensors(&packing);
        let values = reduction.values(&tensors);
        for (claim, &value) in reduction.reductions().iter().zip(&values) {
            assert_eq!(value, bit_evaluation(&packing, claim.point()));
        }
        let mut prover = challenger::<F>();
        let (proof, prover_weights) = reduction.prove(&tensors, &values, &mut prover).unwrap();
        assert_eq!(proof.traces.len(), Coefficients::<EF>::DIMENSION);
        let mut verifier = challenger::<F>();
        let weights = reduction.verify(&values, &proof, &mut verifier).unwrap();
        assert_eq!(weights, prover_weights);
        // The surviving relation holds over the packed vector.
        let table = weights.weight_table();
        let actual: EF = table
            .as_slice()
            .iter()
            .zip(packing.poly().as_slice())
            .map(|(&w, &x)| w * EF::from(x))
            .sum();
        assert_eq!(actual, weights.sum());
        // The evaluator reads the table's multilinear extension.
        let mut rng = StdRng::seed_from_u64(seed ^ 0xabcd);
        let z = Point::<EF>::rand(&mut rng, packed_variables);
        assert_eq!(weights.weight_at(&z), table.eval_base(&z));
        let vertex = Point::<EF>::hypercube(3, packed_variables);
        assert_eq!(weights.weight_at(&vertex), table.as_slice()[3]);
    }

    #[test]
    fn reduces_claims_to_one_weighted_relation_over_gf2_128() {
        round_trip::<BinaryField128, BinaryField128>(5, 1);
    }

    #[test]
    fn reduces_claims_with_a_narrow_packing_and_a_wide_challenge_field() {
        round_trip::<BinaryField8, BinaryField128>(6, 2);
    }

    #[test]
    fn a_wrong_value_is_caught() {
        type F = BinaryField128;
        let (packing, reduction) = fixture::<F, F>(5, 3);
        let tensors = reduction.tensors(&packing);
        let mut values = reduction.values(&tensors);
        values[1] += F::ONE;
        let mut prover = challenger::<F>();
        // The prover computes honest messages for the wrong statement: some has trace one.
        let (proof, _) = reduction
            .prove(&tensors, &values, &mut prover)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut verifier = challenger::<F>();
        let result = reduction.verify(&values, &proof, &mut verifier);
        assert!(
            matches!(result, Err(TraceError::TraceOne { .. })),
            "{result:?}"
        );
        // Shifting the offending message to trace zero moves the batched relation off the witness.
        let mut shifted = proof.clone();
        let basis = TraceBasis::<F>::new();
        for z in shifted.traces.iter_mut() {
            if basis.trace(*z) {
                *z += basis.rho();
            }
        }
        let mut verifier = challenger::<F>();
        let weights = reduction.verify(&values, &shifted, &mut verifier).unwrap();
        let actual: F = weights
            .weight_table()
            .as_slice()
            .iter()
            .zip(packing.poly().as_slice())
            .map(|(&w, &x)| w * x)
            .sum();
        assert_ne!(actual, weights.sum());
    }

    #[test]
    fn refuses_mismatched_claims() {
        type F = BinaryField128;
        let mut rng = StdRng::seed_from_u64(4);
        let a = BitRingSwitch::<F>::new(&Point::<F>::rand(&mut rng, 10)).unwrap();
        let b = BitRingSwitch::<F>::new(&Point::<F>::rand(&mut rng, 11)).unwrap();
        assert_eq!(
            TraceReduction::new(vec![a, b]).err(),
            Some(TraceError::Arity {
                claim: 1,
                expected: 3,
                actual: 4
            })
        );
        assert_eq!(
            TraceReduction::<F>::new(Vec::new()).err(),
            Some(TraceError::NoClaims)
        );
    }

    /// Timing of a root-sized batch: run with `--ignored --nocapture`.
    #[test]
    #[ignore = "timing only"]
    fn times_a_root_sized_batch() {
        type F = BinaryField128;
        let packed_variables = 20;
        let claims = 47;
        let mut rng = StdRng::seed_from_u64(11);
        let num_variables = packed_variables + BitRingSwitch::<F>::ABSORBED;
        let bits: Vec<u8> = (0..1usize << (num_variables - 3))
            .map(|_| rng.random::<u8>())
            .collect();
        let packing = BitPacking::<F>::new(&bits).unwrap();
        let reductions = (0..claims)
            .map(|_| BitRingSwitch::<F>::new(&Point::<F>::rand(&mut rng, num_variables)).unwrap())
            .collect();
        let reduction = TraceReduction::new(reductions).unwrap();
        let start = std::time::Instant::now();
        let tensors = reduction.tensors(&packing);
        let tensors_time = start.elapsed();
        let values = reduction.values(&tensors);
        let start = std::time::Instant::now();
        let (proof, _) = reduction
            .prove(&tensors, &values, &mut challenger::<F>())
            .unwrap();
        let prove_time = start.elapsed();
        let start = std::time::Instant::now();
        let weights = reduction
            .verify(&values, &proof, &mut challenger::<F>())
            .unwrap();
        let verify_time = start.elapsed();
        let z = Point::<F>::rand(&mut rng, packed_variables);
        let start = std::time::Instant::now();
        let _ = weights.weight_at(&z);
        let weight_time = start.elapsed();
        let start = std::time::Instant::now();
        let _ = weights.weight_table();
        let table_time = start.elapsed();
        std::println!(
            "claims {claims} cube 2^{packed_variables}: tensors {tensors_time:?} prove {prove_time:?} verify {verify_time:?} weight_at {weight_time:?} weight_table {table_time:?} proof {} B",
            proof.traces.len() * 16
        );
    }
}
