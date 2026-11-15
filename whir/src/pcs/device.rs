//! A WHIR opening's dense work on a device: the polynomial and its weights stay there from the
//! claims to the final round, every codeword with its Merkle tree beside them.
//!
//! The host plays the transcript, and the device answers what it asks: values, round
//! polynomials, commitments, and the rows and digests a query opens.

use alloc::boxed::Box;

use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::DenseMatrix;
use p3_matrix::extension::FlatMatrixView;
use p3_multilinear_util::point::Point;
use p3_sumcheck::commit::CodewordCommitter;
use p3_sumcheck::layout::StackedDevice;
use p3_sumcheck::ring_switch::bits::PackingRoundsDevice;

/// A round device, borrowed for `'d`.
pub type Device<'d, F, EF, MT> = dyn RoundDevice<F, EF, MT> + 'd;

/// The Merkle prover data of a folded round's codeword.
pub type ExtProverData<F, EF, MT> =
    <MT as Mmcs<F>>::ProverData<FlatMatrixView<F, EF, DenseMatrix<EF>>>;

/// The Merkle prover data of a committed message's codeword.
pub type BaseProverData<F, MT> = <MT as Mmcs<F>>::ProverData<DenseMatrix<F>>;

/// Where a lone commitment's opening runs: a device its message is handed to.
pub trait SingleDevices<F: Field, EF: ExtensionField<F>, MT: Mmcs<F>> {
    /// The rounds over `message`, the lone column of its commitment: its codeword left to the
    /// host that committed it, or, when [`Self::holds_commitment`], the one the device keeps.
    fn single<'a>(&'a self, message: &[F]) -> Box<dyn RoundDevice<F, EF, MT> + 'a>;

    /// Where the bit ring switch ahead of the opening batches its claims' rounds.
    fn packing_rounds(&self) -> &dyn PackingRoundsDevice;

    /// The committer that encodes and commits a lone column here, its message and codeword
    /// then kept for [`Self::single`], or `None` when the host commits.
    fn committer(&self) -> Option<&dyn CodewordCommitter<F, MT>> {
        None
    }

    /// Whether the device holds the commitment it made last, whose queries
    /// [`RoundDevice::first_tree`] then answers.
    fn holds_commitment(&self) -> bool {
        false
    }

    /// The pair openings over the commitment it made last, beside a second commitment, when
    /// it opens pairs.
    fn pair_devices(&self) -> Option<&dyn PairDevices<F, EF, MT>> {
        None
    }
}

/// Where a pair opening runs: a device that holds the first commitment's message and codeword.
pub trait PairDevices<F: Field, EF: ExtensionField<F>, MT: Mmcs<F>> {
    /// One pair opening over `first`, the message this device committed last, and `second`,
    /// each the lone column of its commitment.
    fn pair<'a>(&'a self, first: &[F], second: &[F]) -> Box<dyn PairDevice<F, EF, MT> + 'a>;

    /// [`Self::pair`] over the message this device committed and holds, of `second`'s length,
    /// whose values never reach the host.
    ///
    /// # Panics
    ///
    /// Panics unless the device holds the message it committed last.
    fn pair_held<'a>(&'a self, second: &[F]) -> Box<dyn PairDevice<F, EF, MT> + 'a> {
        let _ = second;
        panic!("this device holds no committed message")
    }

    /// Where the bit ring switches ahead of the pair batch their claims' rounds.
    fn packing_rounds(&self) -> &dyn PackingRoundsDevice;
}

/// One pair opening on a device: the two messages, then their combination.
pub trait PairDevice<F: Field, EF: ExtensionField<F>, MT: Mmcs<F>> {
    /// `f(z_0)`, `g(z_1)`, `g(z_0)` and `f(z_1)`, `f` the first message, `g` the second, and
    /// `z_0`, `z_1` the two points.
    fn values(&mut self, points: &[Point<EF>; 2]) -> [EF; 4];

    /// Replaces the pair by `f + gamma g`.
    fn combine(&mut self, gamma: F);

    /// The combination, to open.
    fn rounds(&mut self) -> &mut dyn RoundDevice<F, EF, MT>;
}

/// The WHIR rounds over a polynomial and its weights held on a device.
///
/// Every point lists its coordinates most significant variable first.
pub trait RoundDevice<F: Field, EF: ExtensionField<F>, MT: Mmcs<F>> {
    /// The polynomial and its weights, as the claims and the sumcheck rounds see them.
    fn stacked(&mut self) -> &mut dyn StackedDevice<F, EF>;

    /// Encodes the polynomial as it stands, `2^folding` evaluations to a row, at rate
    /// `2^-log_inv_rate`, and commits the codeword's rows.
    ///
    /// The codeword and its tree stay on the device under the returned handle.
    fn commit(&mut self, folding: usize, log_inv_rate: usize) -> (MT::Commitment, usize);

    /// The prover data of the codeword under `handle`, holding the rows at `indices` and every
    /// digest their openings read.
    fn tree(&mut self, handle: usize, indices: &[usize]) -> ExtProverData<F, EF, MT>;

    /// The prover data of the first commitment, a pair's or a lone column's, whose codeword the
    /// device kept, holding the rows at `indices` and every digest their openings read.
    fn first_tree(&mut self, indices: &[usize]) -> BaseProverData<F, MT>;

    /// Adds `sum_i c_i eq(z_i, x)` over the `(z_i, c_i)` of `eq`, and
    /// `sum_j c_j prod_k (x_k ? p_jk : 1)` over the `(p_j, c_j)` of `select`, to the weights.
    fn constrain(&mut self, eq: &[(Point<EF>, EF)], select: &[(Point<F>, EF)]);
}
