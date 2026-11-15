//! whir-gc experiment: values of the column-sumcheck closing, split by degree into slots.
//!
//! The closing check is a polynomial of degree at most three in a table's opened values `v`.
//! A [`Polar`] carries one such polynomial split by degree, each part a multilinear form
//! over consecutive *slots*, every slot reading one copy of `v`:
//!
//! ```text
//!     P(v) = c + L(v) + Q(v, v) + T(v, v, v)
//!
//!     c       the constant part
//!     l[s]    the degree-one part, its factor read in slot s           s = 0, 1, 2
//!     q[s]    the degree-two part, its factors read in slots s, s + 1  s = 0, 1
//!     t       the degree-three part, read in slots 0, 1, 2
//! ```
//!
//! A column is the form reading its own entry of each slot's vector.  A product places its
//! left factor's slots before its right factor's:
//!
//! ```text
//!     (a b).q[s] = a.c b.q[s] + a.l[s] b.l[s + 1] + a.q[s] b.c
//! ```
//!
//! so every part stays multilinear, one factor a slot, and agrees with `P` when every slot
//! holds `v`.  That is all the closing sumcheck needs; it never symmetrizes, so
//! characteristic two is no obstacle.
//!
//! A part above degree three has no slot to go to: a product that would make one panics.
//!
//! A [`PolarRoot`] keeps only the parts that start at slot zero.  It is what an AIR's
//! constraints accumulate into: as the left factor of a product it needs nothing more,
//! and the closing reads nothing more.

use core::fmt;
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing};

/// Which parts a value holds, so that structural zeros cost nothing.
const C: u8 = 1;
const L: [u8; 3] = [2, 4, 8];
const Q: [u8; 2] = [16, 32];
const T: u8 = 64;

/// The structural degree of a set of parts.
const fn degree(mask: u8) -> usize {
    if mask & T != 0 {
        3
    } else if mask & (Q[0] | Q[1]) != 0 {
        2
    } else if mask & (L[0] | L[1] | L[2]) != 0 {
        1
    } else {
        0
    }
}

/// A polynomial of the opened values, split by degree into slots.  See the module docs.
pub struct Polar<F, EF> {
    c: EF,
    l: [EF; 3],
    q: [EF; 2],
    t: EF,
    mask: u8,
    _base: PhantomData<fn() -> F>,
}

impl<F, EF: Copy> Clone for Polar<F, EF> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, EF: Copy> Copy for Polar<F, EF> {}

impl<F, EF: fmt::Debug> fmt::Debug for Polar<F, EF> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Polar")
            .field("c", &self.c)
            .field("l", &self.l)
            .field("q", &self.q)
            .field("t", &self.t)
            .field("mask", &self.mask)
            .finish()
    }
}

impl<F, EF: Field> Default for Polar<F, EF> {
    fn default() -> Self {
        Self::ZERO_POLAR
    }
}

impl<F, EF: Field> Polar<F, EF> {
    const ZERO_POLAR: Self = Self {
        c: EF::ZERO,
        l: [EF::ZERO, EF::ZERO, EF::ZERO],
        q: [EF::ZERO, EF::ZERO],
        t: EF::ZERO,
        mask: 0,
        _base: PhantomData,
    };

    /// A constant.
    #[must_use]
    pub const fn constant(c: EF) -> Self {
        Self {
            c,
            mask: C,
            ..Self::ZERO_POLAR
        }
    }

    /// A column, reading `slots[s]` in slot `s`.
    #[must_use]
    pub const fn column(slots: [EF; 3]) -> Self {
        Self {
            l: slots,
            mask: L[0] | L[1] | L[2],
            ..Self::ZERO_POLAR
        }
    }

    /// Every part times `k`.
    #[must_use]
    pub fn scale(self, k: EF) -> Self {
        let mut out = self;
        if self.mask & C != 0 {
            out.c = self.c * k;
        }
        for s in 0..3 {
            if self.mask & L[s] != 0 {
                out.l[s] = self.l[s] * k;
            }
        }
        for s in 0..2 {
            if self.mask & Q[s] != 0 {
                out.q[s] = self.q[s] * k;
            }
        }
        if self.mask & T != 0 {
            out.t = self.t * k;
        }
        out
    }

    fn combine(self, rhs: Self, op: impl Fn(EF, EF) -> EF) -> Self {
        let mask = self.mask | rhs.mask;
        let pick = |bit: u8, a: EF, b: EF| -> EF {
            match (self.mask & bit != 0, rhs.mask & bit != 0) {
                (true, true) => op(a, b),
                (true, false) => a,
                (false, true) => op(EF::ZERO, b),
                (false, false) => EF::ZERO,
            }
        };
        Self {
            c: pick(C, self.c, rhs.c),
            l: core::array::from_fn(|s| pick(L[s], self.l[s], rhs.l[s])),
            q: core::array::from_fn(|s| pick(Q[s], self.q[s], rhs.q[s])),
            t: pick(T, self.t, rhs.t),
            mask,
            _base: PhantomData,
        }
    }

    fn product(self, rhs: Self) -> Self {
        assert!(
            degree(self.mask) + degree(rhs.mask) <= 3,
            "a closing polynomial above degree three"
        );
        let (a, b) = (self, rhs);
        let has = |x: &Self, bit: u8| x.mask & bit != 0;
        let mut out = Self::ZERO_POLAR;
        let add = |out: &mut Self, bit: u8, slot: &mut EF, x: EF| {
            *slot = if out.mask & bit != 0 { *slot + x } else { x };
            out.mask |= bit;
        };
        if has(&a, C) && has(&b, C) {
            let x = a.c * b.c;
            let mut slot = out.c;
            add(&mut out, C, &mut slot, x);
            out.c = slot;
        }
        for s in 0..3 {
            let mut slot = out.l[s];
            if has(&a, C) && has(&b, L[s]) {
                add(&mut out, L[s], &mut slot, a.c * b.l[s]);
            }
            if has(&a, L[s]) && has(&b, C) {
                add(&mut out, L[s], &mut slot, a.l[s] * b.c);
            }
            out.l[s] = slot;
        }
        for s in 0..2 {
            let mut slot = out.q[s];
            if has(&a, C) && has(&b, Q[s]) {
                add(&mut out, Q[s], &mut slot, a.c * b.q[s]);
            }
            if has(&a, L[s]) && has(&b, L[s + 1]) {
                add(&mut out, Q[s], &mut slot, a.l[s] * b.l[s + 1]);
            }
            if has(&a, Q[s]) && has(&b, C) {
                add(&mut out, Q[s], &mut slot, a.q[s] * b.c);
            }
            out.q[s] = slot;
        }
        let mut slot = out.t;
        if has(&a, C) && has(&b, T) {
            add(&mut out, T, &mut slot, a.c * b.t);
        }
        if has(&a, L[0]) && has(&b, Q[1]) {
            add(&mut out, T, &mut slot, a.l[0] * b.q[1]);
        }
        if has(&a, Q[0]) && has(&b, L[2]) {
            add(&mut out, T, &mut slot, a.q[0] * b.l[2]);
        }
        if has(&a, T) && has(&b, C) {
            add(&mut out, T, &mut slot, a.t * b.c);
        }
        out.t = slot;
        out
    }
}

impl<F, EF: Field> Add for Polar<F, EF> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        self.combine(rhs, |a, b| a + b)
    }
}

impl<F, EF: Field> AddAssign for Polar<F, EF> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F, EF: Field> Sub for Polar<F, EF> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self.combine(rhs, |a, b| a - b)
    }
}

impl<F, EF: Field> SubAssign for Polar<F, EF> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F, EF: Field> Neg for Polar<F, EF> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::ZERO_POLAR - self
    }
}

impl<F, EF: Field> Mul for Polar<F, EF> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.product(rhs)
    }
}

impl<F, EF: Field> MulAssign for Polar<F, EF> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F, EF: Field> Sum for Polar<F, EF> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO_POLAR, |a, b| a + b)
    }
}

impl<F, EF: Field> Product for Polar<F, EF> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::constant(EF::ONE), |a, b| a * b)
    }
}

impl<F, EF: Field> PrimeCharacteristicRing for Polar<F, EF> {
    type PrimeSubfield = EF::PrimeSubfield;

    const ZERO: Self = Self::ZERO_POLAR;
    const ONE: Self = Self::constant(EF::ONE);
    const TWO: Self = Self::constant(EF::TWO);
    const NEG_ONE: Self = Self::constant(EF::NEG_ONE);

    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self::constant(EF::from_prime_subfield(f))
    }
}

impl<F: Field, EF: ExtensionField<F>> From<F> for Polar<F, EF> {
    fn from(f: F) -> Self {
        Self::constant(EF::from(f))
    }
}

impl<F: Field, EF: ExtensionField<F>> Add<F> for Polar<F, EF> {
    type Output = Self;
    fn add(self, rhs: F) -> Self {
        self + Self::from(rhs)
    }
}

impl<F: Field, EF: ExtensionField<F>> AddAssign<F> for Polar<F, EF> {
    fn add_assign(&mut self, rhs: F) {
        *self = *self + rhs;
    }
}

impl<F: Field, EF: ExtensionField<F>> Sub<F> for Polar<F, EF> {
    type Output = Self;
    fn sub(self, rhs: F) -> Self {
        self - Self::from(rhs)
    }
}

impl<F: Field, EF: ExtensionField<F>> SubAssign<F> for Polar<F, EF> {
    fn sub_assign(&mut self, rhs: F) {
        *self = *self - rhs;
    }
}

impl<F: Field, EF: ExtensionField<F>> Mul<F> for Polar<F, EF> {
    type Output = Self;
    fn mul(self, rhs: F) -> Self {
        self.scale(EF::from(rhs))
    }
}

impl<F: Field, EF: ExtensionField<F>> MulAssign<F> for Polar<F, EF> {
    fn mul_assign(&mut self, rhs: F) {
        *self = *self * rhs;
    }
}

impl<F: Field, EF: ExtensionField<F>> Algebra<F> for Polar<F, EF> {}

/// The parts of a [`Polar`] that start at slot zero: `[c, l[0], q[0], t]`.
///
/// It multiplies only as a left factor, or by a constant, which is all an accumulator does.
pub struct PolarRoot<F, EF> {
    parts: Polar<F, EF>,
}

impl<F, EF: Copy> Clone for PolarRoot<F, EF> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, EF: Copy> Copy for PolarRoot<F, EF> {}

impl<F, EF: fmt::Debug> fmt::Debug for PolarRoot<F, EF> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.parts.fmt(f)
    }
}

impl<F, EF: Field> Default for PolarRoot<F, EF> {
    fn default() -> Self {
        Self::ZERO
    }
}

/// The parts a root keeps.
const ROOT: u8 = C | L[0] | Q[0] | T;

impl<F, EF: Field> PolarRoot<F, EF> {
    const fn project(parts: Polar<F, EF>) -> Self {
        let mut parts = parts;
        parts.mask &= ROOT;
        Self { parts }
    }

    /// A constant.
    #[must_use]
    pub const fn constant(c: EF) -> Self {
        Self::project(Polar::constant(c))
    }

    /// Every part times `k`.
    #[must_use]
    pub fn scale(self, k: EF) -> Self {
        Self::project(self.parts.scale(k))
    }

    /// `[c, L(s0), Q(s0, s1), T(s0, s1, s2)]`, zero where the root holds no part.
    #[must_use]
    pub fn parts(&self) -> [EF; 4] {
        let pick = |bit: u8, x: EF| if self.parts.mask & bit != 0 { x } else { EF::ZERO };
        [
            pick(C, self.parts.c),
            pick(L[0], self.parts.l[0]),
            pick(Q[0], self.parts.q[0]),
            pick(T, self.parts.t),
        ]
    }

    /// Whether the root is a constant, so that it multiplies from either side.
    const fn is_constant(&self) -> bool {
        self.parts.mask & !C == 0
    }
}

impl<F, EF: Field> From<Polar<F, EF>> for PolarRoot<F, EF> {
    fn from(parts: Polar<F, EF>) -> Self {
        Self::project(parts)
    }
}

impl<F, EF: Field> Add for PolarRoot<F, EF> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::project(self.parts + rhs.parts)
    }
}

impl<F, EF: Field> AddAssign for PolarRoot<F, EF> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F, EF: Field> Sub for PolarRoot<F, EF> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::project(self.parts - rhs.parts)
    }
}

impl<F, EF: Field> SubAssign for PolarRoot<F, EF> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F, EF: Field> Neg for PolarRoot<F, EF> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::project(-self.parts)
    }
}

impl<F, EF: Field> Mul for PolarRoot<F, EF> {
    type Output = Self;
    /// A root holds no part past slot zero, so one side must be a constant.
    fn mul(self, rhs: Self) -> Self {
        if rhs.is_constant() {
            return Self::project(self.parts * rhs.parts);
        }
        assert!(self.is_constant(), "two roots multiply only when one is a constant");
        Self::project(rhs.parts * self.parts)
    }
}

impl<F, EF: Field> MulAssign for PolarRoot<F, EF> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F, EF: Field> Sum for PolarRoot<F, EF> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<F, EF: Field> Product for PolarRoot<F, EF> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |a, b| a * b)
    }
}

impl<F, EF: Field> PrimeCharacteristicRing for PolarRoot<F, EF> {
    type PrimeSubfield = EF::PrimeSubfield;

    const ZERO: Self = Self::project(Polar::ZERO_POLAR);
    const ONE: Self = Self::constant(EF::ONE);
    const TWO: Self = Self::constant(EF::TWO);
    const NEG_ONE: Self = Self::constant(EF::NEG_ONE);

    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self::constant(EF::from_prime_subfield(f))
    }
}

impl<F, EF: Field> Add<Polar<F, EF>> for PolarRoot<F, EF> {
    type Output = Self;
    fn add(self, rhs: Polar<F, EF>) -> Self {
        self + Self::from(rhs)
    }
}

impl<F, EF: Field> AddAssign<Polar<F, EF>> for PolarRoot<F, EF> {
    fn add_assign(&mut self, rhs: Polar<F, EF>) {
        *self = *self + rhs;
    }
}

impl<F, EF: Field> Sub<Polar<F, EF>> for PolarRoot<F, EF> {
    type Output = Self;
    fn sub(self, rhs: Polar<F, EF>) -> Self {
        self - Self::from(rhs)
    }
}

impl<F, EF: Field> SubAssign<Polar<F, EF>> for PolarRoot<F, EF> {
    fn sub_assign(&mut self, rhs: Polar<F, EF>) {
        *self = *self - rhs;
    }
}

impl<F, EF: Field> Mul<Polar<F, EF>> for PolarRoot<F, EF> {
    type Output = Self;
    /// The root is the left factor: it reads the value's parts past slot zero, which it has.
    fn mul(self, rhs: Polar<F, EF>) -> Self {
        Self::project(self.parts * rhs)
    }
}

impl<F, EF: Field> MulAssign<Polar<F, EF>> for PolarRoot<F, EF> {
    fn mul_assign(&mut self, rhs: Polar<F, EF>) {
        *self = *self * rhs;
    }
}

impl<F, EF: Field> Algebra<Polar<F, EF>> for PolarRoot<F, EF> {}

#[cfg(test)]
mod tests {
    use p3_binary_field::BinaryField128;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type EF = BinaryField128;
    type P = Polar<EF, EF>;

    /// A polynomial of three values, of degree three, through the ring operations.
    fn poly<R: PrimeCharacteristicRing + Copy>(x: [R; 3], k: R) -> R {
        let [a, b, c] = x;
        k * (a * b - c) * b + a * a + (R::ONE - c) * (b + k) + a * b * c
    }

    #[test]
    fn every_slot_at_the_values_is_the_polynomial() {
        let mut rng = SmallRng::seed_from_u64(7);
        let v: [EF; 3] = core::array::from_fn(|_| rng.random());
        let k: EF = rng.random();
        let columns: [P; 3] = core::array::from_fn(|i| P::column([v[i]; 3]));
        let value = poly(columns, P::constant(k));
        let root = PolarRoot::from(value).parts();
        assert_eq!(root.iter().copied().sum::<EF>(), poly(v, k));
    }

    #[test]
    fn each_part_is_linear_in_each_slot() {
        // T(s0, s1, s2) at s0 = x + y is T at x plus T at y, holding the other slots.
        let mut rng = SmallRng::seed_from_u64(9);
        let k: EF = rng.random();
        let slots = |s0: [EF; 3], s1: [EF; 3], s2: [EF; 3]| -> [EF; 4] {
            let columns: [P; 3] = core::array::from_fn(|i| P::column([s0[i], s1[i], s2[i]]));
            PolarRoot::from(poly(columns, P::constant(k))).parts()
        };
        let draw = |rng: &mut SmallRng| -> [EF; 3] { core::array::from_fn(|_| rng.random()) };
        let (x, y, s1, s2) = (draw(&mut rng), draw(&mut rng), draw(&mut rng), draw(&mut rng));
        let sum: [EF; 3] = core::array::from_fn(|i| x[i] + y[i]);
        let (at_x, at_y, at_sum) = (slots(x, s1, s2), slots(y, s1, s2), slots(sum, s1, s2));
        for part in 1..4 {
            assert_eq!(at_sum[part], at_x[part] + at_y[part], "part {part}");
        }
        // The constant reads no slot.
        assert_eq!(at_sum[0], at_x[0]);
    }

    #[test]
    #[should_panic(expected = "above degree three")]
    fn a_fourth_degree_panics() {
        let a = P::column([EF::ONE; 3]);
        let _ = a * a * a * a;
    }
}
