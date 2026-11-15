//! Copy constraints between the committed ports of an interleaved composition.
//!
//! A port is one committed slot of a gadget, read by other gadgets' ports.  Every committed
//! word is a cell, the cell of a word its index, so a port's cell is the word it holds.
//!
//! A permutation `sigma` of the cells cycles every set of ports that must hold one word and
//! fixes every other cell.  The words agree around every cycle exactly when the multisets
//!
//! ```text
//! { (c, v_c) }   and   { (sigma(c), v_c) }
//! ```
//!
//! agree, which one product reduction checks over the fingerprints
//! `offset + label * L(c) + v_c`.
//!
//! - `v_c` is the cell's word read at the relation check's within-word point `r`.  Two
//!   different words agree there with probability at most `bits / |F|`, and the words were
//!   committed before `r` was drawn.
//! - `L(c)` is the cell index's bits combined at a sampled label point, distinct for distinct
//!   cells as polynomials in that point.
//!
//! The product's identity leaf claim is `offset + label * L(s) + V(s)` with
//! `V(s) = sum_c eq(s, c) v_c`, the committed trace at `(s, r)`: the reduction leaves one trace
//! opening at the product point itself.
//!
//! The permuted labels `L(sigma)(s)` are one evaluation of the routing's own words: the word of
//! cell `c` is `sigma(c)`, and the label point is a within-word point.  The key commits those
//! words once, and the proof opens that commitment at `(s, tau)`.

mod transcript;

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::Gf2;
use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{Algebra, ExtensionField, Field};
use p3_gkr::{
    DeviceProductTrees, ProductGkrError, ProductGkrProof, ProductGkrRootShape, ProductGkrShape,
};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_word::{Composition, Word};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info_span;
use transcript::{PortChallenges, PortProverTranscript, PortShape, PortVerifierTranscript};

use crate::device::{DevicePorts, WordDevice, from_device, slice_to_device, to_device};
use crate::shift::{byte_tables, evaluate_bytes};

/// One port of one instance of one call.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PortRef {
    /// The call.
    pub call: usize,
    /// The instance of the call.
    pub instance: usize,
    /// The port, an index into the call's port slots.
    pub port: usize,
}

/// A routing the ports of a composition cannot carry.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum PortError {
    /// Ports are laid out over interleaved blocks only.
    #[error("the composition is not interleaved")]
    NotInterleaved,
    /// The port lists do not name one list per call.
    #[error("{lists} port lists for {calls} calls")]
    PortLists {
        /// Lists supplied.
        lists: usize,
        /// Calls in the composition.
        calls: usize,
    },
    /// A port slot is not a committed slot of its call.
    #[error("port slot {slot} of call {call} is past its {slots} committed slots")]
    UnknownSlot {
        /// The call.
        call: usize,
        /// The rejected slot.
        slot: u32,
        /// The call's committed slots.
        slots: usize,
    },
    /// A link names a port that does not exist.
    #[error("port {0:?} does not exist")]
    UnknownPort(PortRef),
    /// The committed words outgrow the cells' label space.
    #[error("the committed words outgrow the cells' label space")]
    TooManyCells,
    /// Two linked ports hold different words.
    #[error("cell {cell} holds another word than the cell it is linked to")]
    Unequal {
        /// The first cell of the disagreeing pair.
        cell: usize,
    },
    /// The device's identity and permuted products differ: two linked ports hold different
    /// words.
    #[error("the identity and permuted fingerprint products differ")]
    UnequalProducts,
    /// The device holds no committed words of the witness's length.
    #[error("the device holds no committed words of {0} words")]
    DeviceWords(usize),
}

/// The committed words a port reduction reads.
#[derive(Clone, Copy)]
pub(crate) enum PortWitness<'a> {
    /// The words, on the host.
    Host(&'a [u64]),
    /// The words a device holds, this many of them.
    Device(&'a dyn WordDevice, usize),
}

/// A port reduction's device trees as the product reduction drives them.
struct TowerPorts<'a, EF> {
    ports: &'a mut dyn DevicePorts,
    layer: tracing::Span,
    _field: core::marker::PhantomData<fn() -> EF>,
}

impl<EF: Field> DeviceProductTrees<EF> for TowerPorts<'_, EF> {
    fn roots(&self) -> Vec<EF> {
        self.ports.roots().into_iter().map(from_device).collect()
    }

    fn binary_children(&self, depth: usize) -> Vec<[EF; 2]> {
        self.ports
            .binary_children(depth)
            .into_iter()
            .map(|children| children.map(from_device))
            .collect()
    }

    fn begin_layer(&mut self, depth: usize, point: &[EF]) {
        self.layer = info_span!("port layer", depth, rounds = point.len());
        self.layer
            .in_scope(|| self.ports.begin_layer(depth, &slice_to_device(point)));
    }

    fn round(&self, nodes: &[EF; 5], batching: EF) -> [EF; 5] {
        let device = |value: EF| to_device(value).expect("only the device's field reaches the device");
        self.layer.in_scope(|| {
            self.ports.round(&nodes.map(device), device(batching)).map(from_device)
        })
    }

    fn fold(&mut self, challenge: EF) {
        let challenge = to_device(challenge).expect("only the device's field reaches the device");
        self.layer.in_scope(|| self.ports.fold(challenge));
    }

    fn children(&self) -> Vec<[EF; 4]> {
        self.ports
            .children()
            .into_iter()
            .map(|children| children.map(from_device))
            .collect()
    }
}

/// A port reduction the verifier refuses.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum PortProofError {
    /// The product reduction failed.
    #[error("product reduction failed: {0}")]
    Product(ProductGkrError),
    /// The product reduction did not return its two leaf values.
    #[error("the product reduction returned {0} leaf values, expected two")]
    LeafValues(usize),
    /// The two leaf values disagree on the port values.
    #[error("the permuted leaf does not differ from the identity leaf by the label difference")]
    Labels,
    /// The committed words outgrow the cells' label space.
    #[error("{0} word variables outgrow the cells' label space")]
    TooManyCells(usize),
}

/// The record of one port reduction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PortProof<F, EF> {
    /// Product reduction of the identity and the permuted fingerprints.
    product: ProductGkrProof<EF>,
    /// Evaluation claimed for the committed routing words at the product point and the label
    /// point.
    permuted_label: EF,
    _field: core::marker::PhantomData<F>,
}

/// What a port reduction leaves to the openings: a trace claim and a routing claim, each a
/// point over committed words followed by a within-word point, and its value.
pub(crate) struct PortClaims<EF> {
    /// The committed trace's claim.
    pub(crate) trace: (Vec<EF>, EF),
    /// The committed routing words' claim.
    pub(crate) routing: (Vec<EF>, EF),
}

/// The copy constraints of one composition's ports, fixed with its key.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PortRouting {
    /// The cells `sigma` moves, ascending, each with its successor on its cycle; `sigma` fixes
    /// every other cell.
    moved: Vec<(u32, u32)>,
}

/// The cells' labels read 32 bits of the cell index.
const LABEL_BITS: usize = 32;

impl PortRouting {
    /// Takes the ports `ports[call]`, committed slots of each call, and cycles every set of
    /// ports `links` joins.
    ///
    /// # Errors
    ///
    /// Returns an error when the composition is not interleaved, a port slot or a linked port
    /// does not exist, or a port's word is past the cells' label space.
    pub fn new<W: Word>(
        composition: &Composition<W>,
        ports: &[Vec<u32>],
        links: &[(PortRef, PortRef)],
    ) -> Result<Self, PortError> {
        if composition.placement() != p3_word::Placement::Interleaved {
            return Err(PortError::NotInterleaved);
        }
        if ports.len() != composition.calls().len() {
            return Err(PortError::PortLists {
                lists: ports.len(),
                calls: composition.calls().len(),
            });
        }
        for (call, slots) in ports.iter().enumerate() {
            let local = composition.calls()[call].component().local_slots();
            if let Some(&slot) = slots.iter().find(|&&slot| slot as usize >= local) {
                return Err(PortError::UnknownSlot {
                    call,
                    slot,
                    slots: local,
                });
            }
        }

        let mut blocks_of = vec![Vec::new(); composition.calls().len()];
        for block in composition.blocks() {
            blocks_of[block.call()].push(block);
        }
        let cell = |port: PortRef| -> Result<u32, PortError> {
            let slot = *ports
                .get(port.call)
                .and_then(|slots| slots.get(port.port))
                .ok_or(PortError::UnknownPort(port))?;
            let block = blocks_of[port.call]
                .iter()
                .find(|block| {
                    (block.first_instance()..block.first_instance() + block.instances())
                        .contains(&port.instance)
                })
                .ok_or(PortError::UnknownPort(port))?;
            let word = block.witness_base()
                + ((slot as usize) << block.log_instances())
                + port.instance
                - block.first_instance();
            u32::try_from(word).map_err(|_| PortError::TooManyCells)
        };

        let mut ends = Vec::with_capacity(links.len());
        for &(a, b) in links {
            ends.push((cell(a)?, cell(b)?));
        }
        let mut cells: Vec<u32> = ends.iter().flat_map(|&(a, b)| [a, b]).collect();
        cells.sort_unstable();
        cells.dedup();
        let index = |cell: u32| cells.binary_search(&cell).expect("a linked cell") as u32;
        let mut parent = (0..cells.len() as u32).collect::<Vec<_>>();
        const fn root(parent: &mut [u32], mut cell: u32) -> u32 {
            while parent[cell as usize] != cell {
                parent[cell as usize] = parent[parent[cell as usize] as usize];
                cell = parent[cell as usize];
            }
            cell
        }
        for &(a, b) in &ends {
            let (a, b) = (root(&mut parent, index(a)), root(&mut parent, index(b)));
            if a != b {
                parent[a.max(b) as usize] = a.min(b);
            }
        }

        let mut first = vec![u32::MAX; cells.len()];
        let mut last = vec![u32::MAX; cells.len()];
        let mut next = (0..cells.len() as u32).collect::<Vec<_>>();
        for member in 0..cells.len() as u32 {
            let class = root(&mut parent, member) as usize;
            if first[class] == u32::MAX {
                first[class] = member;
            } else {
                next[last[class] as usize] = member;
            }
            last[class] = member;
        }
        for class in 0..cells.len() {
            if first[class] != u32::MAX {
                next[last[class] as usize] = first[class];
            }
        }
        let moved = next
            .iter()
            .enumerate()
            .filter(|&(member, &successor)| member as u32 != successor)
            .map(|(member, &successor)| (cells[member], cells[successor as usize]))
            .collect();
        Ok(Self { moved })
    }

    /// The number of cells `sigma` moves.
    #[must_use]
    pub const fn moved(&self) -> usize {
        self.moved.len()
    }

    /// The routing words: the successor of every cell as one word, over `2^word_variables`
    /// words.
    pub(crate) fn words<W: Word>(&self, word_variables: usize) -> Vec<W> {
        let mut words = (0..1_u64 << word_variables)
            .into_par_iter()
            .map(W::from_low_bits)
            .collect::<Vec<_>>();
        for &(cell, next) in &self.moved {
            words[cell as usize] = W::from_low_bits(u64::from(next));
        }
        words
    }

    const fn shape(&self, bit_variables: usize, word_variables: usize) -> PortShape {
        PortShape {
            bit_variables,
            label_variables: bit_variables,
            word_variables,
            moved: self.moved.len(),
        }
    }

    /// The label weight of each word bit, least significant first.
    fn label_weights<EF: Field>(challenges: &PortChallenges<EF>) -> Vec<EF> {
        Point::new(challenges.label_point.as_slice()).equality_weights_msb()
    }

    /// The labels of the cells, by the four bytes of the cell index.
    fn label_tables<EF: Field>(weights: &[EF]) -> [[EF; 256]; 4] {
        core::array::from_fn(|byte| {
            core::array::from_fn(|value| {
                (0..8)
                    .filter(|bit| (value >> bit) & 1 == 1)
                    .map(|bit| weights[8 * byte + bit])
                    .sum()
            })
        })
    }

    /// The product leaf of `point` over the identity labels: the labels' multilinear
    /// extension, the cell bits being the point's coordinates.
    fn identity_label<EF: Field>(weights: &[EF], point: &[EF]) -> EF {
        let n = point.len();
        weights
            .iter()
            .take(n)
            .enumerate()
            .map(|(bit, &weight)| weight * point[n - 1 - bit])
            .sum()
    }

    /// Proves the copy constraints of the committed words `witness`, padded to
    /// `2^word_variables`, with the port values read at `bit_point`.
    ///
    /// Returns the record and the trace opening claim it leaves, at the product point followed
    /// by `bit_point`.
    ///
    /// # Errors
    ///
    /// Returns an error when two linked ports hold different words, or the words outgrow the
    /// cells' label space.
    pub(crate) fn prove<F, EF, Challenger>(
        &self,
        witness: PortWitness<'_>,
        word_variables: usize,
        bit_point: &[EF],
        challenger: &mut Challenger,
    ) -> Result<(PortProof<F, EF>, PortClaims<EF>), PortError>
    where
        F: TranscriptField,
        EF: ExtensionField<F> + Algebra<Gf2>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        if word_variables > LABEL_BITS {
            return Err(PortError::TooManyCells);
        }
        let witness = match witness {
            PortWitness::Host(witness) => witness,
            PortWitness::Device(device, witness_len) => {
                return self.prove_on::<F, EF, _>(device, witness_len, word_variables, bit_point, challenger);
            }
        };
        let word_of = |cell: u32| witness.get(cell as usize).copied().unwrap_or(0);
        if let Some(&(cell, _)) = self.moved.iter().find(|&&(cell, next)| word_of(cell) != word_of(next)) {
            return Err(PortError::Unequal { cell: cell as usize });
        }

        let mut transcript = PortProverTranscript::<_, F, EF>::new(
            challenger,
            self.shape(bit_point.len(), word_variables),
            bit_point,
        );
        let challenges = transcript.challenges(bit_point.len());
        let weights = Self::label_weights(&challenges);

        let cells = 1_usize << word_variables;
        let tables = byte_tables(&Point::new(bit_point).equality_weights_msb());
        let values = (0..cells)
            .into_par_iter()
            .map(|cell| witness.get(cell).map_or(EF::ZERO, |&word| evaluate_bytes(word, &tables)))
            .collect::<Vec<EF>>();
        let labels = Self::label_tables(&weights);
        let label_of = |cell: u32| -> EF {
            (0..4).map(|byte| labels[byte][(cell >> (8 * byte)) as usize & 0xff]).sum()
        };
        let fingerprint =
            |label: u32, value: EF| challenges.offset + challenges.label * label_of(label) + value;
        let identity = (0..cells)
            .into_par_iter()
            .map(|cell| fingerprint(cell as u32, values[cell]))
            .collect::<Vec<_>>();
        let mut permuted = identity.clone();
        for &(cell, next) in &self.moved {
            permuted[cell as usize] = fingerprint(next, values[cell as usize]);
        }
        let shape = ProductGkrShape::new(word_variables, 2, ProductGkrRootShape::FirstTwoShared)
            .expect("a trace's word count is a valid product height");
        let (product, output) = transcript.product(|challenger| {
            ProductGkrProof::prove::<F, _>(&[&identity, &permuted], shape, challenger)
        });
        drop((identity, permuted));

        let point = output.point.as_slice();
        let equality = Point::new(point).equality_weights_msb();
        let permuted_label = Self::identity_label(&weights, point)
            + self
                .moved
                .par_iter()
                .map(|&(cell, next)| equality[cell as usize] * (label_of(next) - label_of(cell)))
                .sum::<EF>();
        transcript.permuted_label(permuted_label);
        let trace_evaluation = equality
            .par_iter()
            .zip(values.par_iter())
            .map(|(&weight, &value)| weight * value)
            .sum::<EF>();

        Ok((
            PortProof {
                product,
                permuted_label,
                _field: core::marker::PhantomData,
            },
            PortClaims {
                trace: (point.iter().chain(bit_point).copied().collect(), trace_evaluation),
                routing: (
                    point.iter().chain(&challenges.label_point).copied().collect(),
                    permuted_label,
                ),
            },
        ))
    }

    /// [`Self::prove`] over the committed words `device` holds, `witness_len` of them.
    fn prove_on<F, EF, Challenger>(
        &self,
        device: &dyn WordDevice,
        witness_len: usize,
        word_variables: usize,
        bit_point: &[EF],
        challenger: &mut Challenger,
    ) -> Result<(PortProof<F, EF>, PortClaims<EF>), PortError>
    where
        F: TranscriptField,
        EF: ExtensionField<F> + Algebra<Gf2>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let device_value =
            |value: EF| to_device(value).expect("only the device's field reaches the device");
        let mut transcript = PortProverTranscript::<_, F, EF>::new(
            challenger,
            self.shape(bit_point.len(), word_variables),
            bit_point,
        );
        let challenges = transcript.challenges(bit_point.len());
        let weights = Self::label_weights(&challenges);
        let (values, labels, moved) = info_span!("port tables").in_scope(|| {
            let values: Vec<u128> = byte_tables(&Point::new(bit_point).equality_weights_msb())
                .iter()
                .flat_map(|table| table.iter().map(|&value| device_value(value)))
                .collect();
            let labels: Vec<u128> = Self::label_tables(&weights)
                .iter()
                .flat_map(|table| table.iter().map(|&value| device_value(value)))
                .collect();
            let moved: Vec<u32> =
                self.moved.iter().flat_map(|&(cell, next)| [cell, next]).collect();
            (values, labels, moved)
        });
        let trees_span = info_span!("port trees").entered();
        let mut ports = device
            .port_trees(
                witness_len,
                word_variables,
                &values,
                &labels,
                device_value(challenges.offset),
                device_value(challenges.label),
                &moved,
            )
            .ok_or(PortError::DeviceWords(witness_len))?;
        let [identity, permuted] = ports.roots();
        drop(trees_span);
        if identity != permuted {
            return Err(PortError::UnequalProducts);
        }
        let shape = ProductGkrShape::new(word_variables, 2, ProductGkrRootShape::FirstTwoShared)
            .expect("a trace's word count is a valid product height");
        let (product, output) = info_span!("port gkr").in_scope(|| {
            transcript.product(|challenger| {
                let mut trees = TowerPorts::<EF> {
                    ports: &mut *ports,
                    layer: tracing::Span::none(),
                    _field: core::marker::PhantomData,
                };
                ProductGkrProof::prove_on::<F, _>(&mut trees, shape, challenger)
            })
        });

        let point = output.point.as_slice();
        let [trace_evaluation, moved_labels] = info_span!("port sums")
            .in_scope(|| ports.sums(&slice_to_device(point)).map(from_device::<EF>));
        info_span!("port release").in_scope(|| drop(ports));
        let permuted_label = Self::identity_label(&weights, point) + moved_labels;
        transcript.permuted_label(permuted_label);

        Ok((
            PortProof {
                product,
                permuted_label,
                _field: core::marker::PhantomData,
            },
            PortClaims {
                trace: (point.iter().chain(bit_point).copied().collect(), trace_evaluation),
                routing: (
                    point.iter().chain(&challenges.label_point).copied().collect(),
                    permuted_label,
                ),
            },
        ))
    }

    /// Verifies a port reduction and returns the unauthenticated trace and routing claims it
    /// leaves.
    ///
    /// # Errors
    ///
    /// Returns an error when the record does not reduce the copy constraints.
    pub(crate) fn verify<F, EF, Challenger>(
        &self,
        proof: &PortProof<F, EF>,
        word_variables: usize,
        bit_point: &[EF],
        challenger: &mut Challenger,
    ) -> Result<PortClaims<EF>, PortProofError>
    where
        F: TranscriptField,
        EF: ExtensionField<F> + Algebra<Gf2>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        if word_variables > LABEL_BITS {
            return Err(PortProofError::TooManyCells(word_variables));
        }
        let mut transcript = PortVerifierTranscript::<_, F, EF>::new(
            challenger,
            self.shape(bit_point.len(), word_variables),
            bit_point,
        );
        let challenges = transcript.challenges(bit_point.len());
        let weights = Self::label_weights(&challenges);
        let shape = ProductGkrShape::new(word_variables, 2, ProductGkrRootShape::FirstTwoShared)
            .expect("a trace's word count is a valid product height");
        let output = match transcript.product(|challenger| proof.product.verify::<F, _>(shape, challenger)) {
            Ok(output) => output,
            Err(error) => {
                transcript.abort();
                return Err(PortProofError::Product(error));
            }
        };
        let [identity, permuted] = output.values[..] else {
            transcript.abort();
            return Err(PortProofError::LeafValues(output.values.len()));
        };
        let point = output.point.as_slice();
        let identity_label = Self::identity_label(&weights, point);
        transcript.permuted_label(proof.permuted_label);
        if identity - permuted != challenges.label * (identity_label - proof.permuted_label) {
            return Err(PortProofError::Labels);
        }
        let trace_evaluation = identity - challenges.offset - challenges.label * identity_label;
        Ok(PortClaims {
            trace: (point.iter().chain(bit_point).copied().collect(), trace_evaluation),
            routing: (
                point.iter().chain(&challenges.label_point).copied().collect(),
                proof.permuted_label,
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;
    use alloc::vec;

    use p3_binary_field::{BinaryChallenger, BinaryField128, TowerLevel};
    use p3_challenger::{CanSample, HashChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;

    use super::*;
    use crate::{DeviceShift, ShiftLayout, Words};

    type F = BinaryField128;
    type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

    fn challenger() -> Challenger {
        Challenger::from_hasher(Vec::new(), Keccak256Hash)
    }

    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// The device contract's port trees, kept on the host.
    struct HostPorts {
        words: Vec<u64>,
        values: Vec<[F; 256]>,
        labels: Vec<[F; 256]>,
        moved: Vec<(u32, u32)>,
        layers: Vec<[Vec<F>; 2]>,
        rows: [Vec<F>; 2],
        point: Vec<F>,
        round: usize,
        equality: Vec<F>,
        prefix: F,
    }

    impl HostPorts {
        fn label(&self, cell: u32) -> F {
            (0..4).map(|byte| self.labels[byte][(cell >> (8 * byte)) as usize & 0xff]).sum()
        }

        fn value(&self, cell: usize) -> F {
            self.words
                .get(cell)
                .map_or(F::ZERO, |&word| evaluate_bytes(word, &self.values))
        }
    }

    impl DevicePorts for HostPorts {
        fn roots(&self) -> [u128; 2] {
            let top = self.layers.last().unwrap();
            [top[0][0].to_repr(), top[1][0].to_repr()]
        }

        fn binary_children(&self, depth: usize) -> [[u128; 2]; 2] {
            let layer = &self.layers[depth];
            [0, 1].map(|t| [layer[t][0].to_repr(), layer[t][1].to_repr()])
        }

        fn begin_layer(&mut self, depth: usize, point: &[u128]) {
            self.rows = self.layers[depth].clone();
            self.point = point.iter().map(|&x| F::from_repr(x)).collect();
            self.round = 0;
            self.equality = Point::new(self.point.get(1..).unwrap_or(&[])).equality_weights_lsb();
            self.prefix = F::ONE;
        }

        fn round(&self, nodes: &[u128; 5], batching: u128) -> [u128; 5] {
            let batching = F::from_repr(batching);
            let anchor = self.point[self.round];
            let pair = [self.prefix - self.prefix * anchor, self.prefix * anchor];
            nodes.map(|node| {
                let x = F::from_repr(node);
                let mut inner = F::ZERO;
                let mut power = F::ONE;
                for rows in &self.rows {
                    let mut sum = F::ZERO;
                    for (j, &weight) in self.equality.iter().enumerate() {
                        let product: F = (0..4)
                            .map(|c| {
                                let (a, b) = (rows[8 * j + c], rows[8 * j + 4 + c]);
                                a + x * (b - a)
                            })
                            .product();
                        sum += weight * product;
                    }
                    inner += power * sum;
                    power *= batching;
                }
                ((pair[0] + x * (pair[1] - pair[0])) * inner).to_repr()
            })
        }

        fn fold(&mut self, challenge: u128) {
            let r = F::from_repr(challenge);
            for rows in &mut self.rows {
                let half = rows.len() / 2;
                let folded: Vec<F> = (0..half)
                    .map(|i| {
                        let (j, c) = (i / 4, i % 4);
                        let (a, b) = (rows[8 * j + c], rows[8 * j + 4 + c]);
                        a + r * (b - a)
                    })
                    .collect();
                *rows = folded;
            }
            let anchor = self.point[self.round];
            let pair = [self.prefix - self.prefix * anchor, self.prefix * anchor];
            self.prefix = pair[0] + r * (pair[1] - pair[0]);
            self.round += 1;
            if self.equality.len() > 1 {
                self.equality = self.equality.chunks(2).map(|p| p[0] + p[1]).collect();
            }
        }

        fn children(&self) -> [[u128; 4]; 2] {
            [0, 1].map(|t| core::array::from_fn(|c| self.rows[t][c].to_repr()))
        }

        fn sums(&self, point: &[u128]) -> [u128; 2] {
            let point: Vec<F> = point.iter().map(|&x| F::from_repr(x)).collect();
            let equality = Point::new(point.as_slice()).equality_weights_msb();
            let value: F = equality.iter().enumerate().map(|(c, &e)| e * self.value(c)).sum();
            let moved: F = self
                .moved
                .iter()
                .map(|&(cell, next)| equality[cell as usize] * (self.label(next) - self.label(cell)))
                .sum();
            [value.to_repr(), moved.to_repr()]
        }
    }

    struct HostDevice(Vec<u64>);

    impl WordDevice for HostDevice {
        fn power_tree(
            &self,
            _: &[u128],
            _: Words<'_>,
            _: Words<'_>,
            _: usize,
            _: usize,
            _: usize,
        ) -> Box<dyn crate::DeviceTree> {
            unreachable!("the port reduction builds no word tree")
        }

        fn bit_tree(
            &self,
            _: &[Words<'_>],
            _: &[u128],
            _: usize,
            _: usize,
            _: usize,
        ) -> Box<dyn crate::DeviceTree> {
            unreachable!("the port reduction builds no word tree")
        }

        fn bit_values(&self, _: &[Words<'_>], _: usize, _: &[u128]) -> Vec<u128> {
            unreachable!("the port reduction reads no bit values")
        }

        fn selector(
            &self,
            _: &[Vec<u128>],
            _: &[u128],
            _: Words<'_>,
            _: &[u128],
            _: Words<'_>,
            _: usize,
            _: [u128; 3],
        ) -> Box<dyn crate::DeviceSelector> {
            unreachable!("the port reduction builds no selector")
        }

        fn relation(
            &self,
            _: &[Words<'_>],
            _: &[Vec<u128>],
            _: usize,
        ) -> Box<dyn crate::DeviceRelation> {
            unreachable!("the port reduction checks no relation")
        }

        fn shift(
            &self,
            _: Words<'_>,
            _: usize,
            _: usize,
            _: u64,
            _: &dyn Fn() -> ShiftLayout,
            _: &[u128],
            _: &[u128],
            _: &[u128],
        ) -> Box<dyn DeviceShift> {
            unreachable!("the port reduction runs no shift")
        }

        fn gather_witness(&self, _: &[u32]) -> bool {
            unreachable!("the test gathers no words")
        }

        fn port_trees(
            &self,
            witness_len: usize,
            word_variables: usize,
            values: &[u128],
            labels: &[u128],
            offset: u128,
            label: u128,
            moved: &[u32],
        ) -> Option<Box<dyn DevicePorts>> {
            if witness_len != self.0.len() {
                return None;
            }
            let tables = |flat: &[u128]| -> Vec<[F; 256]> {
                flat.chunks(256).map(|t| core::array::from_fn(|i| F::from_repr(t[i]))).collect()
            };
            let mut ports = HostPorts {
                words: self.0.clone(),
                values: tables(values),
                labels: tables(labels),
                moved: moved.chunks(2).map(|p| (p[0], p[1])).collect(),
                layers: Vec::new(),
                rows: [Vec::new(), Vec::new()],
                point: Vec::new(),
                round: 0,
                equality: Vec::new(),
                prefix: F::ONE,
            };
            let (offset, label) = (F::from_repr(offset), F::from_repr(label));
            let cells = 1_usize << word_variables;
            let identity: Vec<F> = (0..cells)
                .map(|c| offset + label * ports.label(c as u32) + ports.value(c))
                .collect();
            let mut permuted = identity.clone();
            for &(cell, next) in &ports.moved {
                permuted[cell as usize] = offset + label * ports.label(next) + ports.value(cell as usize);
            }
            let mut layers = vec![[identity, permuted]];
            while layers.last().unwrap()[0].len() > 1 {
                let next = layers.last().unwrap().each_ref().map(|layer| {
                    layer.chunks(2).map(|p| p[0] * p[1]).collect::<Vec<F>>()
                });
                layers.push(next);
            }
            ports.layers = layers;
            Some(Box::new(ports))
        }
    }

    #[test]
    fn device_port_trees_reach_the_host_proof() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for (word_variables, witness_len) in [(1_usize, 2_usize), (2, 3), (5, 29), (6, 64), (7, 100)] {
            let mut words: Vec<u64> = (0..witness_len).map(|_| next(&mut state)).collect();
            let mut moved = Vec::new();
            let cycles = witness_len / 3;
            for k in 0..cycles {
                let (a, b, c) = (3 * k, 3 * k + 1, 3 * k + 2);
                words[b] = words[a];
                words[c] = words[a];
                moved.extend([(a as u32, b as u32), (b as u32, c as u32), (c as u32, a as u32)]);
            }
            let routing = PortRouting { moved };
            let bit_point: Vec<F> = (0..6)
                .map(|_| F::from_repr(u128::from(next(&mut state)) << 64 | u128::from(next(&mut state))))
                .collect();

            let mut host_challenger = challenger();
            let (host, host_claims) = routing
                .prove::<F, F, _>(PortWitness::Host(&words), word_variables, &bit_point, &mut host_challenger)
                .unwrap();
            let device = HostDevice(words.clone());
            let mut device_challenger = challenger();
            let (on, on_claims) = routing
                .prove::<F, F, _>(
                    PortWitness::Device(&device, witness_len),
                    word_variables,
                    &bit_point,
                    &mut device_challenger,
                )
                .unwrap();
            assert_eq!(
                postcard::to_allocvec(&host).unwrap(),
                postcard::to_allocvec(&on).unwrap(),
                "{word_variables} variables"
            );
            assert_eq!(host_claims.trace, on_claims.trace);
            assert_eq!(host_claims.routing, on_claims.routing);
            assert_eq!(
                CanSample::<F>::sample(&mut host_challenger),
                CanSample::<F>::sample(&mut device_challenger)
            );

            let mut verifier = challenger();
            let claims = routing.verify::<F, F, _>(&on, word_variables, &bit_point, &mut verifier).unwrap();
            assert_eq!(claims.trace, on_claims.trace);
        }
    }
}
