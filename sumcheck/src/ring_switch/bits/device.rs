//! The batched packing rounds of a bit ring switch on a device.
//!
//! The batched reduction proves one sum against the packing's slot:
//!
//! ```text
//!     sum_w t'(w) * W(w),    W = sum_i lambda^i * A_i
//! ```
//!
//! A device builds `W` next to the slot it was handed and plays the product rounds over the
//! two, the host keeping only the transcript. Every value crossing is the tower-basis bit
//! pattern of an element of `GF(2^128)`.
//!
//! Each claim's weights over its run of the slot are a coordinate map of an equality product,
//! shifted along the columns when the claim sends successor elements:
//!
//! ```text
//!     e(i)   = outer[i >> k] * inner[i mod 2^k]
//!     A(i)   = sum_b table_b[byte b of e(i)]
//!     W(i)  += A(i) + [i mod c != 0] * alpha * A(i - 1) + [i mod c = c - 1] * alpha^2 * A(i)
//! ```
//!
//! The rounds bind the leading variable first: a round sends, over the halves of both tables,
//!
//! ```text
//!     c_a = sum t'_lo * W_lo,    c_inf = sum (t'_hi - t'_lo) * (W_hi - W_lo)
//! ```
//!
//! and its challenge `r` folds both, `lo + r * (hi - lo)`.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::any::Any;

use p3_binary_field::{BinaryField128, TowerLevel};
use p3_field::{Field, PrimeCharacteristicRing, RawDataSerializable};

/// What builds the packing rounds on a device.
pub trait PackingRoundsDevice {
    /// The rounds over `packing`, the slot's elements, every weight zero.
    fn begin<'a>(&'a self, packing: &[u128]) -> Box<dyn PackingRounds + 'a>;

    /// [`Self::begin`] over the `len` elements from `offset` of the message this device
    /// committed and holds.
    ///
    /// # Panics
    ///
    /// Panics unless the device holds the committed message.
    fn begin_held<'a>(&'a self, offset: usize, len: usize) -> Box<dyn PackingRounds + 'a> {
        let _ = (offset, len);
        panic!("this device holds no committed message")
    }

    /// The elements a claim sends over the held message, by their columns, with
    /// `e(w) = outer[w >> k] * inner[w mod 2^k]` over the `outer.len() << k` elements from
    /// `offset`, `inner` holding `2^k`, and `t(w) = t'(offset + w)`:
    ///
    /// ```text
    ///     tensor[v] = sum_w e(w) * bit_v(t(w))
    ///     carry[v]  = sum_{w : (w + 1) mod c != 0} e(w) * bit_v(t(w + 1))
    ///     last[v]   = sum_{w : w mod c = c - 1} e(w) * bit_v(t(w))
    /// ```
    ///
    /// the tensor's 128 columns, then, when the claim's columns are `successor = Some(c)`
    /// elements long, the carry's and the last element's.
    ///
    /// # Panics
    ///
    /// Panics unless the device holds the committed message.
    fn held_tensor(&self, offset: usize, outer: &[u128], inner: &[u128], successor: Option<usize>) -> Vec<u128> {
        let _ = (offset, outer, inner, successor);
        panic!("this device holds no committed message")
    }
}

/// The weights and rounds of one batched reduction, held on a device.
pub trait PackingRounds {
    /// Adds a claim's weights to the run of `outer.len() * inner.len()` entries from `start`:
    /// `tables` are sixteen tables of 256 entries, the byte positions of a product's pattern
    /// lowest first, and `successor` the column length and challenge of the shift.
    fn add_weights(
        &mut self,
        start: usize,
        outer: &[u128],
        inner: &[u128],
        tables: &[u128],
        successor: Option<(usize, u128)>,
    );

    /// The round's `c_a` and `c_inf`.
    fn round(&mut self) -> [u128; 2];

    /// Binds the leading variable of both tables at `challenge`.
    fn fold(&mut self, challenge: u128);

    /// The packing's value once every variable is bound.
    fn last(&mut self) -> u128;
}

/// The pattern of `value`, an element of `GF(2^128)`.
pub(crate) fn pattern<T: Field>(value: T) -> u128 {
    (&value as &dyn Any)
        .downcast_ref::<BinaryField128>()
        .expect("a packing device runs over GF(2^128)")
        .to_repr()
}

/// The element of `T`, which is `GF(2^128)`, a pattern names.
pub(crate) fn from_pattern<T: Field>(value: u128) -> T {
    *(&BinaryField128::from_repr(value) as &dyn Any)
        .downcast_ref::<T>()
        .expect("a packing device runs over GF(2^128)")
}

/// The patterns of a whole slice.
pub(crate) fn patterns<T: Field>(values: &[T]) -> Vec<u128> {
    values.iter().map(|&value| pattern(value)).collect()
}

/// The device contract kept on the host, for tests of the callers.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostPackingRounds;

struct HostRounds {
    packing: Vec<BinaryField128>,
    weights: Vec<BinaryField128>,
}

impl PackingRoundsDevice for HostPackingRounds {
    fn begin<'a>(&'a self, packing: &[u128]) -> Box<dyn PackingRounds + 'a> {
        Box::new(HostRounds {
            packing: packing.iter().map(|&x| BinaryField128::from_repr(x)).collect(),
            weights: vec![BinaryField128::ZERO; packing.len()],
        })
    }
}

impl PackingRounds for HostRounds {
    fn add_weights(
        &mut self,
        start: usize,
        outer: &[u128],
        inner: &[u128],
        tables: &[u128],
        successor: Option<(usize, u128)>,
    ) {
        let field = |x: &u128| BinaryField128::from_repr(*x);
        let (outer, inner): (Vec<_>, Vec<_>) =
            (outer.iter().map(field).collect(), inner.iter().map(field).collect());
        let tables: Vec<_> = tables.iter().map(field).collect();
        let weight = |i: usize| {
            let e = outer[i / inner.len()] * inner[i % inner.len()];
            e.into_bytes()
                .iter()
                .enumerate()
                .map(|(b, &byte)| tables[256 * b + usize::from(byte)])
                .sum::<BinaryField128>()
        };
        let len = outer.len() * inner.len();
        for i in 0..len {
            let mut value = weight(i);
            if let Some((column, alpha)) = successor {
                let alpha = BinaryField128::from_repr(alpha);
                if i % column != 0 {
                    value += alpha * weight(i - 1);
                }
                if i % column == column - 1 {
                    value += alpha.square() * weight(i);
                }
            }
            self.weights[start + i] += value;
        }
    }

    fn round(&mut self) -> [u128; 2] {
        let half = self.packing.len() / 2;
        let (mut c_a, mut c_inf) = (BinaryField128::ZERO, BinaryField128::ZERO);
        for i in 0..half {
            let (p, q) = (self.packing[i], self.packing[i + half]);
            let (v, w) = (self.weights[i], self.weights[i + half]);
            c_a += p * v;
            c_inf += (q - p) * (w - v);
        }
        [c_a.to_repr(), c_inf.to_repr()]
    }

    fn fold(&mut self, challenge: u128) {
        let r = BinaryField128::from_repr(challenge);
        let half = self.packing.len() / 2;
        for table in [&mut self.packing, &mut self.weights] {
            for i in 0..half {
                table[i] = table[i] + r * (table[i + half] - table[i]);
            }
            table.truncate(half);
        }
    }

    fn last(&mut self) -> u128 {
        self.packing[0].to_repr()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_field::BinaryChallenger;
    use p3_challenger::{CanSample, HashChallenger};
    use p3_keccak::Keccak256Hash;
    use p3_multilinear_util::point::Point;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::ring_switch::bits::{BitPacking, BitRingSwitch, BitRingSwitchClaims};

    type EF = BinaryField128;
    type Chal = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;

    fn challenger() -> Chal {
        Chal::from_hasher(Vec::new(), Keccak256Hash)
    }

    fn point(rng: &mut SmallRng, prefix: &[bool], num_variables: usize) -> Point<EF> {
        Point::new(
            (0..num_variables)
                .map(|i| match prefix.get(i) {
                    Some(&bit) => EF::from_bool(bit),
                    None => rng.random(),
                })
                .collect(),
        )
    }

    /// The held-message contract kept on the host: the message the device would hold.
    struct HostHeld(Vec<u128>);

    impl PackingRoundsDevice for HostHeld {
        fn begin<'a>(&'a self, packing: &[u128]) -> Box<dyn PackingRounds + 'a> {
            HostPackingRounds.begin(packing)
        }

        fn begin_held<'a>(&'a self, offset: usize, len: usize) -> Box<dyn PackingRounds + 'a> {
            HostPackingRounds.begin(&self.0[offset..offset + len])
        }

        fn held_tensor(&self, offset: usize, outer: &[u128], inner: &[u128], successor: Option<usize>) -> Vec<u128> {
            let field = |x: &u128| EF::from_repr(*x);
            let len = outer.len() * inner.len();
            let mut columns = vec![EF::ZERO; if successor.is_some() { 384 } else { 128 }];
            let mut add = |part: usize, e: EF, t: u128| {
                for v in 0..128 {
                    if (t >> v) & 1 == 1 {
                        columns[128 * part + v] += e;
                    }
                }
            };
            for w in 0..len {
                let e = field(&outer[w / inner.len()]) * field(&inner[w % inner.len()]);
                add(0, e, self.0[offset + w]);
                if let Some(c) = successor {
                    if (w + 1) % c != 0 {
                        add(1, e, self.0[offset + w + 1]);
                    } else {
                        add(2, e, self.0[offset + w]);
                    }
                }
            }
            columns.iter().map(|&c| pattern(c)).collect()
        }
    }

    /// Claims by their Boolean prefixes and successor column lengths.
    type Claims<'a> = &'a [(&'a [bool], Option<usize>)];

    #[test]
    fn held_message_reaches_the_host_proof() {
        let mut rng = SmallRng::seed_from_u64(0x4e1d);
        let cases: [(usize, Claims<'_>); 5] = [
            (12, &[(&[], None), (&[], None)]),
            (14, &[(&[true], None), (&[true, false], None), (&[true, true, false], None)]),
            (13, &[(&[false], None), (&[true], None)]),
            (12, &[(&[], None), (&[], Some(9))]),
            (14, &[(&[true], Some(10)), (&[true, false], Some(11)), (&[], Some(12))]),
        ];
        for (num_variables, claims) in cases {
            let witness: Vec<u8> = (0..1usize << (num_variables - 3)).map(|_| rng.random()).collect();
            let packing = BitPacking::<EF>::new(&witness).unwrap();
            let held = HostHeld(packing.poly().as_slice().iter().map(|&x| pattern(x)).collect());
            let setup = BitRingSwitchClaims::new(
                claims
                    .iter()
                    .map(|&(prefix, successor)| {
                        let point = point(&mut rng, prefix, num_variables);
                        successor.map_or_else(
                            || BitRingSwitch::<EF>::new(&point),
                            |rows| BitRingSwitch::<EF>::with_successor(&point, rows),
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap();

            let mut host = challenger();
            let (host_proof, host_point, host_value) = setup.prove::<EF, _, _>(&packing, &mut host);
            let mut on = challenger();
            let (proof, point, value) = setup.prove_held(&held, &mut on);
            assert_eq!(
                serde_json::to_string(&host_proof).unwrap(),
                serde_json::to_string(&proof).unwrap(),
                "{num_variables} variables"
            );
            assert_eq!((host_point, host_value), (point, value));
            assert_eq!(CanSample::<EF>::sample(&mut host), CanSample::<EF>::sample(&mut on));
        }
    }

    #[test]
    fn device_rounds_reach_the_host_proof() {
        let mut rng = SmallRng::seed_from_u64(0x5eed);
        let cases: [(usize, Claims<'_>); 4] = [
            (12, &[(&[], None), (&[], Some(9))]),
            (14, &[(&[], Some(3)), (&[], Some(12)), (&[], None)]),
            (14, &[(&[true], None), (&[true, false], Some(10)), (&[true, true, false], Some(4))]),
            (13, &[(&[false], Some(8)), (&[true], Some(11))]),
        ];
        for (num_variables, claims) in cases {
            let witness: Vec<u8> = (0..1usize << (num_variables - 3)).map(|_| rng.random()).collect();
            let packing = BitPacking::<EF>::new(&witness).unwrap();
            let setup = BitRingSwitchClaims::new(
                claims
                    .iter()
                    .map(|&(prefix, successor)| {
                        let point = point(&mut rng, prefix, num_variables);
                        successor.map_or_else(
                            || BitRingSwitch::<EF>::new(&point),
                            |rows| BitRingSwitch::<EF>::with_successor(&point, rows),
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap();

            let mut host = challenger();
            let (host_proof, host_point, host_value) = setup.prove::<EF, _, _>(&packing, &mut host);
            let mut on = challenger();
            let (proof, point, value) =
                setup.prove_on::<EF, _, _>(&packing, Some(&HostPackingRounds), &mut on);
            assert_eq!(
                serde_json::to_string(&host_proof).unwrap(),
                serde_json::to_string(&proof).unwrap(),
                "{num_variables} variables"
            );
            assert_eq!((host_point, host_value), (point, value));
            assert_eq!(CanSample::<EF>::sample(&mut host), CanSample::<EF>::sample(&mut on));
        }
    }
}
