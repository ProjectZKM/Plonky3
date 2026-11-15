//! whir-gc experiment: a closed pair opening, which sends none of the opened values.
//!
//! The caller's closing check is a polynomial of degree at most three in the values, one
//! polynomial a table (see [`ClosingRelation`]).  A column sumcheck reduces it to each
//! table's columns at three column points, and those are what the bit commitment opens.
//!
//! # The column space
//!
//! A table's main and preprocessed columns share one space of `n_t` coordinates, the
//! preprocessed half above the main half when the table has preprocessed columns:
//!
//! ```text
//!     main column j          ->  index j
//!     preprocessed column j  ->  index 2^(n_t - 1) + j
//! ```
//!
//! Every table is zero above its own space inside the widest one, `M` coordinates, so it
//! reads the trailing `n_t` coordinates of a point under `eq(0, leading)`, as the forms
//! sumcheck reads its tables.  Coordinates are big-endian: the first is the top index bit.
//!
//! # The sumcheck
//!
//! The relation splits table `t`'s polynomial by degree into forms over three slots,
//! `c_t + L_t(s0) + Q_t(s0, s1) + T_t(s0, s1, s2)`.  With `V_t` the table's column
//! polynomial and `delta` the indicator of index zero, three sumchecks of `M` rounds each,
//! one per point, prove
//!
//! ```text
//!     target - sum_t c_t  =  sum_{z0, z1, z2} sum_t V_t(z0) [ L_t(z0) delta(z1) delta(z2)
//!                               + V_t(z1) ( Q_t(z0, z1) delta(z2) + V_t(z2) T_t(z0, z1, z2) ) ]
//! ```
//!
//! Every round has degree two and sends two values.  Once `z0` is bound, the rest is a sum
//! over `z1` and `z2` with `V_t(z0)` a known factor, and likewise for `z1`.  The last round
//! leaves the columns at the three points, which the prover sends as one fold per column
//! block of each side, and the forms at the points' equality tables, which the verifier
//! evaluates once a table.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{BitCoordinates, TowerLevel};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::Table;
use p3_sumcheck::{ClosingRelation, OpeningEvals, OpeningProtocol};

use super::plan::{OpeningRoute, TableRun};
use super::{BooleanTraceCommitment, BooleanTraceCommitmentError};
use crate::boolean::{BitOpening, BitReadings, BooleanBackend};
use crate::boolean_trace_transcript::{
    ClosingSumcheckProverTranscript, ClosingSumcheckShape, ClosingSumcheckVerifierTranscript,
};

/// Coordinates a width's columns need: `ceil(log2(width))`, zero for one column.
const fn bits(width: usize) -> usize {
    width.next_power_of_two().trailing_zeros() as usize
}

/// One side's columns in a table's closing space.
///
/// The bit commitment reads a side's columns in blocks aligned in its slot space, which
/// need not be aligned in column order: a table whose first column sits at an odd slot
/// splits into blocks of different sizes.  The side's index space packs the blocks largest
/// first, which keeps every block aligned in at most one coordinate more than its columns
/// need, so each block's fold is the side's polynomial on one subcube.
#[derive(Clone, Debug)]
pub(super) struct Side {
    /// Each column's index.
    index: Vec<usize>,
    /// Each block's first index, in the run's block order.
    starts: Vec<usize>,
    /// Coordinates of the side's index space.
    variables: usize,
}

impl Side {
    /// The side a run's blocks lay out.
    fn of_run(run: &TableRun) -> Self {
        let sizes: Vec<usize> = run.blocks.iter().map(|block| 1usize << block.variables).collect();
        Self::packed(run.shape.width, &sizes, |b, i| run.blocks[b].first + i)
    }

    /// Columns `0..width` as one block, in column order.
    #[cfg(test)]
    fn contiguous(width: usize) -> Self {
        Self::packed(width, &[1 << bits(width)], |_, i| i)
    }

    /// Blocks of `sizes` slots, packed largest first; `column(b, i)` is block `b`'s `i`-th.
    fn packed(width: usize, sizes: &[usize], column: impl Fn(usize, usize) -> usize) -> Self {
        let mut order: Vec<usize> = (0..sizes.len()).collect();
        order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]));
        let mut starts = vec![0; sizes.len()];
        let mut next = 0;
        for &b in &order {
            starts[b] = next;
            next += sizes[b];
        }
        let mut index = vec![usize::MAX; width];
        for (b, &size) in sizes.iter().enumerate() {
            for i in 0..size {
                let c = column(b, i);
                if c < width {
                    index[c] = starts[b] + i;
                }
            }
        }
        debug_assert!(index.iter().all(|&i| i != usize::MAX), "the blocks cover every column");
        Self {
            index,
            starts,
            variables: bits(next.max(1)),
        }
    }

    const fn width(&self) -> usize {
        self.index.len()
    }
}

/// One table's place in the closing column space.
#[derive(Clone, Debug)]
pub(super) struct Space {
    /// The main side.
    main: Side,
    /// The preprocessed side, when the table has preprocessed columns.
    preprocessed: Option<Side>,
    /// Coordinates of the table's own space.
    variables: usize,
}

impl Space {
    fn new(main: Side, preprocessed: Option<Side>) -> Self {
        let variables = match &preprocessed {
            None => main.variables,
            Some(side) => 1 + main.variables.max(side.variables),
        };
        Self {
            main,
            preprocessed,
            variables,
        }
    }

    /// A table of contiguous sides, `preprocessed` zero for none.
    #[cfg(test)]
    fn contiguous(main: usize, preprocessed: usize) -> Self {
        Self::new(
            Side::contiguous(main),
            (preprocessed > 0).then(|| Side::contiguous(preprocessed)),
        )
    }

    /// Index of the preprocessed half.
    const fn half(&self) -> usize {
        1 << (self.variables - 1)
    }

    /// Each column's index in the table's space, main then preprocessed.
    fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        let half = if self.preprocessed.is_some() { self.half() } else { 0 };
        self.main.index.iter().copied().chain(
            self.preprocessed
                .iter()
                .flat_map(move |side| side.index.iter().map(move |&i| half + i)),
        )
    }

    const fn main_width(&self) -> usize {
        self.main.width()
    }

    /// The main then the preprocessed columns of a vector over the table's space.
    fn split<T: Copy>(&self, vector: &[T]) -> (Vec<T>, Vec<T>) {
        let main_width = self.main_width();
        let mut main = Vec::with_capacity(main_width);
        let mut preprocessed = Vec::new();
        for (column, index) in self.indices().enumerate() {
            if column < main_width {
                main.push(vector[index]);
            } else {
                preprocessed.push(vector[index]);
            }
        }
        (main, preprocessed)
    }

    /// This table's value at a point, from its main and preprocessed side polynomials at
    /// the point's trailing coordinates: the side bit and the gaps above each side's own
    /// coordinates select them.
    fn combine<EF: Field>(&self, tail: &[EF], main: EF, preprocessed: EF) -> EF {
        debug_assert_eq!(tail.len(), self.variables);
        let zeros = |coords: &[EF]| coords.iter().map(|&u| EF::ONE - u).product::<EF>();
        let Some(side) = &self.preprocessed else {
            return main;
        };
        let bit = tail[0];
        let half = &tail[1..];
        let main_gap = zeros(&half[..half.len() - self.main.variables]);
        let preprocessed_gap = zeros(&half[..half.len() - side.variables]);
        (EF::ONE - bit) * main_gap * main + bit * preprocessed_gap * preprocessed
    }
}

/// `eq(point, i)` for every index `i`, big-endian, over the last `k` coordinates of `point`,
/// for each `k` up to `max`: one expansion from the last coordinate up.
fn suffix_eq_tables<EF: Field>(point: &[EF], max: usize) -> Vec<Vec<EF>> {
    let mut tables = Vec::with_capacity(max + 1);
    tables.push(vec![EF::ONE]);
    for k in 1..=max {
        let c = point[point.len() - k];
        let previous: &Vec<EF> = &tables[k - 1];
        let high: Vec<EF> = previous.iter().map(|&x| c * x).collect();
        let mut table: Vec<EF> = previous.iter().zip(&high).map(|(&x, &h)| x - h).collect();
        table.extend(high);
        tables.push(table);
    }
    tables
}

/// The runs of a closed side: one batch each, current row only.
fn closed_runs<E>(protocol: &OpeningProtocol) -> Result<Vec<TableRun>, BooleanTraceCommitmentError<E>> {
    match OpeningRoute::new(protocol) {
        Some(OpeningRoute::Batched(runs))
            if runs
                .iter()
                .all(|run| run.shape.num_batches == 1 && !run.shape.next && run.shape.sparse_next.is_empty()) =>
        {
            Ok(runs)
        }
        _ => Err(BooleanTraceCommitmentError::InvalidOpeningProtocol),
    }
}

/// The two sides' runs matched to the relation's tables: every table has its main run, in
/// order, and the tables with preprocessed columns have theirs, in order.
#[allow(clippy::type_complexity)]
fn match_tables<E>(
    widths: &[(usize, usize)],
    main: &[TableRun],
    preprocessed: &[TableRun],
) -> Result<Vec<(usize, Option<usize>)>, BooleanTraceCommitmentError<E>> {
    if main.len() != widths.len() {
        return Err(BooleanTraceCommitmentError::ClosingSumcheck);
    }
    let mut next_preprocessed = 0;
    let mut matched = Vec::with_capacity(widths.len());
    for (table, &(main_width, preprocessed_width)) in widths.iter().enumerate() {
        if main[table].shape.width != main_width || main_width == 0 {
            return Err(BooleanTraceCommitmentError::ClosingSumcheck);
        }
        let preprocessed_run = if preprocessed_width == 0 {
            None
        } else {
            let run = preprocessed
                .get(next_preprocessed)
                .ok_or(BooleanTraceCommitmentError::ClosingSumcheck)?;
            if run.shape.width != preprocessed_width {
                return Err(BooleanTraceCommitmentError::ClosingSumcheck);
            }
            next_preprocessed += 1;
            Some(next_preprocessed - 1)
        };
        matched.push((table, preprocessed_run));
    }
    if next_preprocessed != preprocessed.len() {
        return Err(BooleanTraceCommitmentError::ClosingSumcheck);
    }
    Ok(matched)
}

/// The closing shape of matched tables.
fn closing_shape(
    widths: &[(usize, usize)],
    main: &[TableRun],
    preprocessed: &[TableRun],
    matched: &[(usize, Option<usize>)],
) -> (ClosingSumcheckShape, Vec<Space>) {
    let spaces: Vec<Space> = matched
        .iter()
        .map(|&(table, p)| {
            Space::new(
                Side::of_run(&main[table]),
                p.map(|p| Side::of_run(&preprocessed[p])),
            )
        })
        .collect();
    let column_variables = spaces.iter().map(|space| space.variables).max().unwrap_or(0);
    let main_folds = 3 * main.iter().map(|run| run.blocks.len()).sum::<usize>();
    let preprocessed_folds = 3 * matched
        .iter()
        .filter_map(|&(_, p)| p)
        .map(|p| preprocessed[p].blocks.len())
        .sum::<usize>();
    (
        ClosingSumcheckShape {
            column_variables,
            main_folds,
            preprocessed_folds,
            widths: widths.to_vec(),
        },
        spaces,
    )
}

/// The trailing `variables` coordinates of `point`.
fn tail<EF: Field>(point: &Point<EF>, variables: usize) -> Point<EF> {
    let coords = point.as_slice();
    Point::new(coords[coords.len() - variables..].to_vec())
}

impl<EF, B> BooleanTraceCommitment<EF, B>
where
    EF: BitCoordinates + ExtensionField<B::Val>,
    B: BooleanBackend<EF>,
    B::Val: TranscriptField + TowerLevel,
{
    /// The folds of one side's run at one column point, with their claims and readings.
    #[allow(clippy::type_complexity)]
    fn closed_side(
        run: &TableRun,
        side: &Side,
        pad: usize,
        row_point: &Point<EF>,
        column_point: &Point<EF>,
        values: Option<&[EF]>,
        folds: Option<&[EF]>,
    ) -> (Vec<EF>, EF, Vec<(BitOpening<EF>, BitReadings<EF>)>) {
        let side_point = tail(column_point, side.variables);
        // Each block's point and its selector in the side's index space.
        let places: Vec<(Point<EF>, EF)> = run
            .blocks
            .iter()
            .zip(&side.starts)
            .map(|(block, &start)| {
                let lead = side.variables - block.variables;
                let place = start >> block.variables;
                let selector = side_point.as_slice()[..lead]
                    .iter()
                    .enumerate()
                    .map(|(i, &u)| if (place >> (lead - 1 - i)) & 1 == 1 { u } else { EF::ONE - u })
                    .product();
                (Point::new(side_point.as_slice()[lead..].to_vec()), selector)
            })
            .collect();
        let folds: Vec<EF> = match (values, folds) {
            (Some(values), _) => run
                .blocks
                .iter()
                .zip(&places)
                .map(|(block, (block_point, _))| {
                    let end = (block.first + (1 << block.variables)).min(values.len());
                    Self::combine_columns(&values[block.first..end], block_point)
                })
                .collect(),
            (None, Some(folds)) => folds.to_vec(),
            (None, None) => unreachable!("a side is either evaluated or read"),
        };
        let value = places
            .iter()
            .zip(&folds)
            .map(|((_, selector), &fold)| *selector * fold)
            .sum();
        let claims = Self::forms_block_claims(run, pad, row_point, places, &folds).collect();
        (folds, value, claims)
    }

    /// Prove a closed pair opening's column sumcheck over both sides' evaluated values.
    ///
    /// Returns the round values, the three column points, and each table's value at each.
    #[allow(clippy::too_many_lines)]
    pub(super) fn prove_closing<Challenger>(
        spaces: &[Space],
        values: &[(Vec<EF>, Vec<EF>)],
        closing: &dyn ClosingRelation<EF>,
        shape: &ClosingSumcheckShape,
        transcript: &mut ClosingSumcheckProverTranscript<'_, B::Val, EF, Challenger>,
    ) -> (Vec<EF>, [Point<EF>; 3])
    where
        Challenger: FieldChallenger<B::Val>
            + GrindingChallenger<Witness = B::Val>
            + CanSampleUniformBits<B::Val>,
    {
        let m = shape.column_variables;
        let size = 1usize << m;
        // Each table's column vector over the widest space, zero above its own.
        let padded: Vec<Vec<EF>> = values
            .iter()
            .zip(spaces)
            .map(|((main, preprocessed), space)| {
                let mut vector = EF::zero_vec(size);
                for (index, &value) in space.indices().zip(main.iter().chain(preprocessed)) {
                    vector[index] = value;
                }
                vector
            })
            .collect();
        let own = |vector: &[EF], space: &Space| -> (Vec<EF>, Vec<EF>) { space.split(vector) };

        let mut rounds = Vec::with_capacity(2 * shape.rounds());
        // The padded equality table of each bound point, and each table's value there.
        let mut tables_at: Vec<Vec<EF>> = Vec::with_capacity(3);
        let mut points: Vec<Point<EF>> = Vec::with_capacity(3);
        let mut at: Vec<Vec<EF>> = Vec::with_capacity(3);
        let mut claim: EF = (0..spaces.len())
            .map(|t| {
                let (main, preprocessed) = &values[t];
                let slot = (main.as_slice(), preprocessed.as_slice());
                let forms = closing.evaluate(t, [slot, slot, slot]);
                forms[1] + forms[2] + forms[3]
            })
            .sum();
        for phase in 0..3 {
            // Each table's coefficient of its columns this phase: its forms with this slot
            // one column at a time, the earlier slots at their bound points and the later
            // ones at the values, times the table's values at the bound points.
            let coefficients: Vec<Vec<EF>> = spaces
                .iter()
                .enumerate()
                .map(|(t, space)| {
                    let scale: EF = at.iter().map(|values_at| values_at[t]).product();
                    let earlier: Vec<(Vec<EF>, Vec<EF>)> =
                        tables_at.iter().map(|table| own(table, space)).collect();
                    let (main, preprocessed) = &values[t];
                    let main_width = space.main_width();
                    let column = |column: usize| -> EF {
                        let mut one_main = EF::zero_vec(main_width);
                        let mut one_preprocessed = EF::zero_vec(preprocessed.len());
                        if column < main_width {
                            one_main[column] = EF::ONE;
                        } else {
                            one_preprocessed[column - main_width] = EF::ONE;
                        }
                        let mut slots: [(&[EF], &[EF]); 3] =
                            [(main.as_slice(), preprocessed.as_slice()); 3];
                        for (slot, (m, p)) in slots.iter_mut().zip(&earlier) {
                            *slot = (m.as_slice(), p.as_slice());
                        }
                        slots[phase] = (one_main.as_slice(), one_preprocessed.as_slice());
                        let forms = closing.evaluate(t, slots);
                        forms[phase + 1..].iter().copied().sum::<EF>() * scale
                    };
                    let indices: Vec<usize> = space.indices().collect();
                    let entries: Vec<EF> =
                        (0..indices.len()).into_par_iter().map(column).collect();
                    let mut coefficient = EF::zero_vec(size);
                    for (&index, entry) in indices.iter().zip(entries) {
                        coefficient[index] = entry;
                    }
                    coefficient
                })
                .collect();
            // The indicator's weight: the lower-degree forms already closed by earlier slots.
            let mut indicator = EF::ZERO;
            if phase > 0 {
                for (t, space) in spaces.iter().enumerate() {
                    let (main, preprocessed) = &values[t];
                    let earlier: Vec<(Vec<EF>, Vec<EF>)> =
                        tables_at.iter().map(|table| own(table, space)).collect();
                    let mut slots: [(&[EF], &[EF]); 3] =
                        [(main.as_slice(), preprocessed.as_slice()); 3];
                    for (slot, (m, p)) in slots.iter_mut().zip(&earlier) {
                        *slot = (m.as_slice(), p.as_slice());
                    }
                    let forms = closing.evaluate(t, slots);
                    let a0 = at[0][t];
                    indicator += if phase == 1 {
                        a0 * forms[1]
                    } else {
                        let zero1: EF = points[1].as_slice().iter().map(|&u| EF::ONE - u).product();
                        a0 * (forms[1] * zero1 + at[1][t] * forms[2])
                    };
                }
            }
            let mut columns: Vec<Vec<EF>> = padded.clone();
            let mut coefficients = coefficients;
            let mut coordinates = Vec::with_capacity(m);
            for _ in 0..m {
                let half = columns[0].len() / 2;
                let (mut low, mut high, mut quadratic) = (EF::ZERO, EF::ZERO, EF::ZERO);
                for (column, coefficient) in columns.iter().zip(&coefficients) {
                    for i in 0..half {
                        low += column[i] * coefficient[i];
                        high += column[half + i] * coefficient[half + i];
                        quadratic +=
                            (column[half + i] - column[i]) * (coefficient[half + i] - coefficient[i]);
                    }
                }
                low += indicator;
                debug_assert_eq!(low + high, claim, "closing round {phase} does not sum to its claim");
                let linear = claim - low.double() - quadratic;
                rounds.push(low);
                rounds.push(quadratic);
                let r = transcript.round([low, quadratic]);
                claim = low + linear * r + quadratic * r.square();
                for vector in columns.iter_mut().chain(coefficients.iter_mut()) {
                    for i in 0..half {
                        let (lo, hi) = (vector[i], vector[half + i]);
                        vector[i] = lo + r * (hi - lo);
                    }
                    vector.truncate(half);
                }
                indicator *= EF::ONE - r;
                coordinates.push(r);
            }
            at.push(columns.iter().map(|column| column[0]).collect());
            let point = Point::new(coordinates);
            tables_at.push(suffix_eq_tables(point.as_slice(), m).pop().expect("the full table"));
            points.push(point);
        }
        let [p0, p1, p2]: [Point<EF>; 3] = points.try_into().expect("three points");
        (rounds, [p0, p1, p2])
    }

    /// Check a closed pair opening's column sumcheck, and return its three column points.
    ///
    /// `folds_at` reads each table's value at each point from the folds, once bound.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(super) fn verify_closing<Challenger>(
        spaces: &[Space],
        closing: &dyn ClosingRelation<EF>,
        shape: &ClosingSumcheckShape,
        rounds: &[EF],
        main_folds: &[EF],
        preprocessed_folds: &[EF],
        transcript: &mut ClosingSumcheckVerifierTranscript<'_, B::Val, EF, Challenger>,
        mut folds_at: impl FnMut(&[Point<EF>; 3]) -> Vec<[EF; 3]>,
    ) -> Result<[Point<EF>; 3], BooleanTraceCommitmentError<B::Error>>
    where
        Challenger: FieldChallenger<B::Val>
            + GrindingChallenger<Witness = B::Val>
            + CanSampleUniformBits<B::Val>,
    {
        let m = shape.column_variables;
        let target = closing.target().ok_or(BooleanTraceCommitmentError::ClosingSumcheck)?;
        let mut challenges = Vec::with_capacity(shape.rounds());
        for round in rounds.chunks_exact(2) {
            challenges.push(transcript.round([round[0], round[1]])?);
        }
        transcript.folds(main_folds, preprocessed_folds)?;
        let points: [Point<EF>; 3] = core::array::from_fn(|p| Point::new(challenges[p * m..(p + 1) * m].to_vec()));
        // Each table's value at each point, read off its folds.
        let values_at = folds_at(&points);

        // Each point's suffix equality tables and `eq(0, point)`.
        let tables: Vec<Vec<Vec<EF>>> = points
            .iter()
            .map(|point| suffix_eq_tables(point.as_slice(), m))
            .collect();
        let zero: [EF; 3] =
            core::array::from_fn(|p| points[p].as_slice().iter().map(|&u| EF::ONE - u).product());

        let mut constant = EF::ZERO;
        let mut expected = EF::ZERO;
        for (t, space) in spaces.iter().enumerate() {
            let n = space.variables;
            let lead: [EF; 3] = core::array::from_fn(|p| {
                points[p].as_slice()[..m - n].iter().map(|&u| EF::ONE - u).product()
            });
            let split: [(Vec<EF>, Vec<EF>); 3] = core::array::from_fn(|p| space.split(&tables[p][n]));
            let slots: [(&[EF], &[EF]); 3] =
                core::array::from_fn(|p| (split[p].0.as_slice(), split[p].1.as_slice()));
            let [c, l, q, cubic] = closing.evaluate(t, slots);
            constant += c;
            // Padded: each table value and each form carries its points' leading zeros.
            let a: [EF; 3] = core::array::from_fn(|p| lead[p] * values_at[t][p]);
            let l = lead[0] * l;
            let q = lead[0] * lead[1] * q;
            let cubic = lead[0] * lead[1] * lead[2] * cubic;
            expected += a[0] * (l * zero[1] * zero[2] + a[1] * (q * zero[2] + a[2] * cubic));
        }

        let mut claim = target - constant;
        for (round, &r) in rounds.chunks_exact(2).zip(&challenges) {
            let (low, quadratic) = (round[0], round[1]);
            let linear = claim - low.double() - quadratic;
            claim = low + linear * r + quadratic * r.square();
        }
        if claim != expected {
            transcript.abort();
            return Err(BooleanTraceCommitmentError::ClosingSumcheck);
        }
        Ok(points)
    }

    /// The prover's side of a closed pair opening before the bit opening: each table's
    /// values, the sumcheck, the folds, and every claim with its reading.
    ///
    /// The main side's values come from `known` where the caller supplies them.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub(super) fn close_pair<Challenger>(
        &self,
        second: &Self,
        protocols: [&OpeningProtocol; 2],
        row_points: [&[Point<EF>]; 2],
        tables: [&[Table<B::Val>]; 2],
        known: &[Option<OpeningEvals<EF>>],
        closing: &dyn ClosingRelation<EF>,
        challenger: &mut Challenger,
    ) -> Result<
        (Vec<EF>, Vec<EF>, [Vec<BitOpening<EF>>; 2], [Vec<BitReadings<EF>>; 2]),
        BooleanTraceCommitmentError<B::Error>,
    >
    where
        Challenger: FieldChallenger<B::Val>
            + GrindingChallenger<Witness = B::Val>
            + CanSampleUniformBits<B::Val>,
    {
        let widths = closing.widths();
        let main_runs = closed_runs(protocols[0])?;
        let preprocessed_runs = closed_runs(protocols[1])?;
        let matched = match_tables(&widths, &main_runs, &preprocessed_runs)?;
        let (shape, spaces) = closing_shape(&widths, &main_runs, &preprocessed_runs, &matched);
        let pads = [
            self.protocol_pad(&protocols[0].table_shapes())?,
            second.protocol_pad(&protocols[1].table_shapes())?,
        ];
        // Each table's values at its row point, before the transcript moves.
        let evaluate = |side: usize, run: &TableRun| -> Vec<EF> {
            let table = &tables[side][run.table];
            assert!(!table.is_held(), "a closed opening evaluates its tables on the host");
            let columns: Vec<usize> = (0..run.shape.width).collect();
            Self::evaluate_requested(table, &row_points[side][run.opening], &columns, &[]).0
        };
        let values: Vec<(Vec<EF>, Vec<EF>)> = matched
            .iter()
            .map(|&(table, preprocessed)| {
                let run = &main_runs[table];
                let main = known
                    .get(run.opening)
                    .and_then(Option::as_ref)
                    .map_or_else(|| evaluate(0, run), |known| known.current().to_vec());
                let preprocessed =
                    preprocessed.map_or_else(Vec::new, |p| evaluate(1, &preprocessed_runs[p]));
                (main, preprocessed)
            })
            .collect();
        let values = values.as_slice();
        let mut transcript = ClosingSumcheckProverTranscript::<B::Val, EF, Challenger>::new(challenger, &shape);
        let (rounds, points) = Self::prove_closing(&spaces, values, closing, &shape, &mut transcript);

        let mut folds = [Vec::new(), Vec::new()];
        let mut openings = [Vec::new(), Vec::new()];
        let mut readings = [Vec::new(), Vec::new()];
        for point in &points {
            for &(table, preprocessed) in &matched {
                let run = &main_runs[table];
                let (f, _, claims) = Self::closed_side(
                    run,
                    &spaces[table].main,
                    pads[0],
                    &row_points[0][run.opening],
                    point,
                    Some(&values[table].0),
                    None,
                );
                folds[0].extend(f);
                for (opening, reading) in claims {
                    openings[0].push(opening);
                    readings[0].push(reading);
                }
                if let Some(p) = preprocessed {
                    let run = &preprocessed_runs[p];
                    let (f, _, claims) = Self::closed_side(
                        run,
                        spaces[table].preprocessed.as_ref().expect("a preprocessed run has its side"),
                        pads[1],
                        &row_points[1][run.opening],
                        point,
                        Some(&values[table].1),
                        None,
                    );
                    folds[1].extend(f);
                    for (opening, reading) in claims {
                        openings[1].push(opening);
                        readings[1].push(reading);
                    }
                }
            }
        }
        transcript.folds(&folds[0], &folds[1]);
        transcript.finish();
        let [main_folds, preprocessed_folds] = folds;
        let mut main_values = rounds;
        main_values.extend(main_folds);
        Ok((main_values, preprocessed_folds, openings, readings))
    }

    /// The verifier's side of [`PrescribedPointPcs::verify_pair_closed`] before the bit
    /// opening: the sumcheck against the relation, and every claim with its reading.
    #[allow(clippy::type_complexity)]
    pub(super) fn check_closed_pair<Challenger>(
        &self,
        second: &Self,
        protocols: [&OpeningProtocol; 2],
        row_points: [&[Point<EF>]; 2],
        proof_values: [&[EF]; 2],
        closing: &dyn ClosingRelation<EF>,
        challenger: &mut Challenger,
    ) -> Result<([Vec<BitOpening<EF>>; 2], [Vec<BitReadings<EF>>; 2]), BooleanTraceCommitmentError<B::Error>>
    where
        Challenger: FieldChallenger<B::Val>
            + GrindingChallenger<Witness = B::Val>
            + CanSampleUniformBits<B::Val>,
    {
        let widths = closing.widths();
        let main_runs = closed_runs(protocols[0])?;
        let preprocessed_runs = closed_runs(protocols[1])?;
        let matched = match_tables(&widths, &main_runs, &preprocessed_runs)?;
        let (shape, spaces) = closing_shape(&widths, &main_runs, &preprocessed_runs, &matched);
        // Every count is checked before the transcript moves.
        if proof_values[0].len() != 2 * shape.rounds() + shape.main_folds
            || proof_values[1].len() != shape.preprocessed_folds
        {
            return Err(BooleanTraceCommitmentError::ClosingSumcheck);
        }
        let pads = [
            self.protocol_pad(&protocols[0].table_shapes())?,
            second.protocol_pad(&protocols[1].table_shapes())?,
        ];
        let (rounds, main_folds) = proof_values[0].split_at(2 * shape.rounds());
        let preprocessed_folds = proof_values[1];

        let mut openings = [Vec::new(), Vec::new()];
        let mut readings = [Vec::new(), Vec::new()];
        let folds_at = |points: &[Point<EF>; 3]| -> Vec<[EF; 3]> {
            let mut values_at = vec![[EF::ZERO; 3]; spaces.len()];
            let mut main_rest = main_folds;
            let mut preprocessed_rest = preprocessed_folds;
            for (p, point) in points.iter().enumerate() {
                for &(table, preprocessed) in &matched {
                    let run = &main_runs[table];
                    let (here, after) = main_rest.split_at(run.blocks.len());
                    main_rest = after;
                    let (_, main_value, claims) = Self::closed_side(
                        run,
                        &spaces[table].main,
                        pads[0],
                        &row_points[0][run.opening],
                        point,
                        None,
                        Some(here),
                    );
                    for (opening, reading) in claims {
                        openings[0].push(opening);
                        readings[0].push(reading);
                    }
                    let mut preprocessed_value = EF::ZERO;
                    if let Some(index) = preprocessed {
                        let run = &preprocessed_runs[index];
                        let (here, after) = preprocessed_rest.split_at(run.blocks.len());
                        preprocessed_rest = after;
                        let (_, value, claims) = Self::closed_side(
                            run,
                            spaces[table].preprocessed.as_ref().expect("a preprocessed run has its side"),
                            pads[1],
                            &row_points[1][run.opening],
                            point,
                            None,
                            Some(here),
                        );
                        preprocessed_value = value;
                        for (opening, reading) in claims {
                            openings[1].push(opening);
                            readings[1].push(reading);
                        }
                    }
                    let space = &spaces[table];
                    let coords = point.as_slice();
                    values_at[table][p] =
                        space.combine(&coords[coords.len() - space.variables..], main_value, preprocessed_value);
                }
            }
            values_at
        };
        let mut transcript =
            ClosingSumcheckVerifierTranscript::<B::Val, EF, Challenger>::new(challenger, &shape);
        let _points = Self::verify_closing(
            &spaces,
            closing,
            &shape,
            rounds,
            main_folds,
            preprocessed_folds,
            &mut transcript,
            folds_at,
        )?;
        transcript.finish();
        Ok((openings, readings))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::boolean::BooleanPcs;
    use crate::test_util::{MyChallenger, MyMmcs, challenger};

    type EF = BinaryField128;
    type Commitment = BooleanTraceCommitment<EF, BooleanPcs<EF, MyMmcs, MyMmcs>>;

    /// A table's cubic: a constant and random terms of each degree over its columns, main
    /// then preprocessed.
    struct Toy {
        widths: Vec<(usize, usize)>,
        constant: Vec<EF>,
        linear: Vec<Vec<(usize, EF)>>,
        quadratic: Vec<Vec<(usize, usize, EF)>>,
        cubic: Vec<Vec<(usize, usize, usize, EF)>>,
        target: Option<EF>,
    }

    fn at(slot: (&[EF], &[EF]), main: usize, index: usize) -> EF {
        if index < main { slot.0[index] } else { slot.1[index - main] }
    }

    impl ClosingRelation<EF> for Toy {
        fn widths(&self) -> Vec<(usize, usize)> {
            self.widths.clone()
        }

        fn target(&self) -> Option<EF> {
            self.target
        }

        fn evaluate(&self, table: usize, slots: [(&[EF], &[EF]); 3]) -> [EF; 4] {
            let main = self.widths[table].0;
            let l = self.linear[table].iter().map(|&(i, k)| k * at(slots[0], main, i)).sum();
            let q = self.quadratic[table]
                .iter()
                .map(|&(i, j, k)| k * at(slots[0], main, i) * at(slots[1], main, j))
                .sum();
            let t = self.cubic[table]
                .iter()
                .map(|&(i, j, h, k)| {
                    k * at(slots[0], main, i) * at(slots[1], main, j) * at(slots[2], main, h)
                })
                .sum();
            [self.constant[table], l, q, t]
        }
    }

    /// A table's column vector over its own space, evaluated at the trailing coordinates of
    /// `point` under the leading zeros, straight from the definition.
    fn value_at(space: &Space, values: &(Vec<EF>, Vec<EF>), point: &[EF]) -> EF {
        let m = point.len();
        let eq = suffix_eq_tables(point, m).pop().unwrap();
        space.indices().zip(values.0.iter().chain(&values.1)).map(|(i, &v)| v * eq[i]).sum()
    }

    #[test]
    fn a_fold_reads_its_columns_in_the_equality_tables_order() {
        let mut rng = SmallRng::seed_from_u64(3);
        let random = |rng: &mut SmallRng| EF::from_u128(rng.random::<u128>());
        for n in 1..6 {
            let values: Vec<EF> = (0..(1 << n) - 1).map(|_| random(&mut rng)).collect();
            let point: Vec<EF> = (0..n).map(|_| random(&mut rng)).collect();
            let eq = suffix_eq_tables(&point, n).pop().unwrap();
            let direct: EF = values.iter().zip(&eq).map(|(&v, &e)| v * e).sum();
            assert_eq!(Commitment::combine_columns(&values, &Point::new(point)), direct, "n = {n}");
        }
    }

    #[test]
    fn the_column_sumcheck_closes_on_a_toy_cubic_and_refuses_a_wrong_target() {
        let mut rng = SmallRng::seed_from_u64(7);
        let widths = vec![(3, 2), (5, 0), (2, 6), (1, 1)];
        let random = |rng: &mut SmallRng| EF::from_u128(rng.random::<u128>());
        let mut toy = Toy {
            widths: widths.clone(),
            constant: vec![],
            linear: vec![],
            quadratic: vec![],
            cubic: vec![],
            target: None,
        };
        for &(main, preprocessed) in &widths {
            let n = main + preprocessed;
            toy.constant.push(random(&mut rng));
            toy.linear.push((0..3).map(|_| (rng.random_range(0..n), random(&mut rng))).collect());
            toy.quadratic.push(
                (0..3)
                    .map(|_| (rng.random_range(0..n), rng.random_range(0..n), random(&mut rng)))
                    .collect(),
            );
            toy.cubic.push(
                (0..3)
                    .map(|_| {
                        (
                            rng.random_range(0..n),
                            rng.random_range(0..n),
                            rng.random_range(0..n),
                            random(&mut rng),
                        )
                    })
                    .collect(),
            );
        }
        let values: Vec<(Vec<EF>, Vec<EF>)> = widths
            .iter()
            .map(|&(main, preprocessed)| {
                (
                    (0..main).map(|_| random(&mut rng)).collect(),
                    (0..preprocessed).map(|_| random(&mut rng)).collect(),
                )
            })
            .collect();
        let target: EF = values
            .iter()
            .enumerate()
            .map(|(t, (main, preprocessed))| {
                let slot = (main.as_slice(), preprocessed.as_slice());
                toy.evaluate(t, [slot; 3]).iter().copied().sum::<EF>()
            })
            .sum();
        let mut spaces: Vec<Space> = widths.iter().map(|&(m, p)| Space::contiguous(m, p)).collect();
        // A main side starting at an odd slot: blocks of two then four columns, packed
        // largest first.
        spaces[2] = Space::new(
            Side::packed(2, &[2], |_, i| i),
            Some(Side::packed(6, &[2, 4], |b, i| if b == 0 { i } else { 2 + i })),
        );
        let shape = ClosingSumcheckShape {
            column_variables: spaces.iter().map(|s| s.variables).max().unwrap(),
            main_folds: 0,
            preprocessed_folds: 0,
            widths: widths.clone(),
        };

        let mut prover_challenger = challenger();
        let mut transcript =
            ClosingSumcheckProverTranscript::<EF, EF, MyChallenger>::new(&mut prover_challenger, &shape);
        let (rounds, points) = Commitment::prove_closing(&spaces, &values, &toy, &shape, &mut transcript);
        transcript.folds(&[], &[]);
        transcript.finish();
        assert_eq!(rounds.len(), 2 * shape.rounds());

        let verify = |toy: &Toy| {
            let mut verifier_challenger = challenger();
            let mut transcript = ClosingSumcheckVerifierTranscript::<EF, EF, MyChallenger>::new(
                &mut verifier_challenger,
                &shape,
            );
            let m = shape.column_variables;
            let result = Commitment::verify_closing(
                &spaces,
                toy,
                &shape,
                &rounds,
                &[],
                &[],
                &mut transcript,
                |points: &[Point<EF>; 3]| {
                    spaces
                        .iter()
                        .zip(&values)
                        .map(|(space, values)| {
                            // The table's own value at the point's tail, without the leading zeros.
                            core::array::from_fn(|p| {
                                let coords = points[p].as_slice();
                                let lead: EF = coords[..m - space.variables]
                                    .iter()
                                    .map(|&u| EF::ONE - u)
                                    .product();
                                let tail = &coords[m - space.variables..];
                                let direct = value_at(space, values, tail);
                                let padded = value_at(space, values, coords);
                                assert_eq!(lead * direct, padded, "the leading zeros factor out");
                                direct
                            })
                        })
                        .collect()
                },
            );
            if result.is_ok() {
                transcript.finish();
            }
            result
        };
        toy.target = Some(target);
        let verified = verify(&toy).expect("an honest column sumcheck closes");
        assert_eq!(verified, points);
        toy.target = Some(target + EF::ONE);
        assert!(verify(&toy).is_err(), "a wrong target is refused");
    }
}
