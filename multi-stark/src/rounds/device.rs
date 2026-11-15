//! The zerocheck's row sums on a device.
//!
//! A stage's tables go to the device once, as dense columns of the field's bit patterns,
//! with the three boundary selectors as columns of their own, so a round's work is the same
//! for every row and every round: for each pair of rows, the high half of the current
//! variable against the low, and each interpolation node, every column is read at the node,
//! each AIR's constraints and interactions are evaluated by a program compiled from its
//! symbolic constraints, and the results are summed with the equality weights.  The host
//! keeps the claims, the degree groups and the transcript, and combines the sums as it does
//! for its own rows.  A fold binds the variable on the device, in place, and reports the
//! wrapped next-row values the host's openings carry.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_air::symbolic::{AirLayout, BaseEntry, BaseLeaf, SymbolicExpr, SymbolicExpression};
use p3_air::{Air, BaseAir, BoundaryEnd};
use p3_field::{ExtensionField, Field};
use p3_lookup::InteractionSymbolicBuilder;

use super::{
    AirOpenings, AirSlot, BoundaryEvals, ExtColumns, RoundStateBase, RoundStateExt,
    evaluated_nodes, finish_round, next_row_runs,
};

/// Read a constant: `a` indexes the program's constants.
pub const OP_CONST: u32 = 0;
/// Read column `a` at the current row.
pub const OP_LOCAL: u32 = 1;
/// Read column `a` at the next row.
pub const OP_NEXT: u32 = 2;
/// The first-row selector.
pub const OP_FIRST: u32 = 3;
/// The last-row selector.
pub const OP_LAST: u32 = 4;
/// The transition selector.
pub const OP_TRANSITION: u32 = 5;
/// `dst = a + b`.
pub const OP_ADD: u32 = 6;
/// `dst = a - b`.
pub const OP_SUB: u32 = 7;
/// `dst = -a`.
pub const OP_NEG: u32 = 8;
/// `dst = a * b`.
pub const OP_MUL: u32 = 9;
/// Constraint `b` is register `a`: add `alpha_powers[b] * a` to the constraint accumulator.
pub const OP_CONSTRAINT: u32 = 10;
/// Field `b` of the pending interaction is register `a`: add `theta_beta_powers[b] * a` to
/// the fingerprint.
pub const OP_FIELD: u32 = 11;
/// Close the pending interaction: its multiplicity is register `a`, its lookup is entry `b`
/// of the slot's link, and weight `dst` of that lookup scales
/// `multiplicity + offset - fingerprint` into the interaction accumulator.
pub const OP_INTERACTION: u32 = 12;

/// One step of a slot's program.  Operands are registers unless the operation says
/// otherwise; `dst` is the register written, when one is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Instruction {
    pub op: u32,
    pub a: u32,
    pub b: u32,
    pub dst: u32,
}

/// An AIR's constraints and interactions as a program over a register file.
#[derive(Clone, Debug)]
pub struct Program<F> {
    pub code: Vec<Instruction>,
    pub registers: usize,
    pub constants: Vec<F>,
}

/// One lookup of a link: its bus offset and the weight of each of its blocks.
#[derive(Clone, Debug)]
pub struct LookupSpec<EF> {
    pub offset: EF,
    pub weights: Vec<EF>,
}

/// One AIR's link: its local lookups first, then its global ones.
#[derive(Clone, Debug)]
pub struct LinkSpec<EF> {
    pub num_local: usize,
    pub lookups: Vec<LookupSpec<EF>>,
}

/// The interaction family of a slot.
#[derive(Clone, Copy, Debug)]
pub struct InteractionSpec {
    pub degree: usize,
    pub group: usize,
    pub link: usize,
}

/// One AIR of the stage.
#[derive(Clone, Debug)]
pub struct SlotSpec<F, EF> {
    pub stage: usize,
    pub constraint_degree: usize,
    pub interaction: Option<InteractionSpec>,
    pub alpha_powers: Vec<EF>,
    pub program: Program<F>,
}

/// Everything a device needs for one stage of the zerocheck.
#[derive(Clone, Debug)]
pub struct StageSpec<F, EF> {
    pub num_vars: usize,
    /// How many columns the slots' programs address.
    pub width: usize,
    /// Each table's instance in the proof, in the stage's table order.
    pub instances: Vec<usize>,
    /// Where each table's columns sit among the addressed columns, in the same order.
    pub tables: Vec<TableSpec>,
    /// The columns whose next row the programs read.
    pub next_columns: Vec<u32>,
    pub slots: Vec<SlotSpec<F, EF>>,
    pub theta_beta_powers: Vec<EF>,
    pub links: Vec<LinkSpec<EF>>,
    /// How many constraint sums a round reports: one per table.
    pub stages: usize,
    /// How many interaction sums a round reports: one per interaction group.
    pub groups: usize,
    /// The point of the stage's row variables after the first, most significant first: a
    /// round weighs each pair of rows by its equality there, and every fold sums out the
    /// leading variable.
    pub eq_point: Vec<EF>,
}

/// Where a table's committed columns sit among a stage's addressed columns.
#[derive(Clone, Copy, Debug)]
pub struct TableSpec {
    pub main_offset: usize,
    pub main_width: usize,
    /// The preprocessed columns' offset and width, when the table has them.
    pub preprocessed: Option<(usize, usize)>,
}

/// A device that takes every stage of the zerocheck: with a device, no stage runs on the host.
pub trait ZerocheckDevice<F, EF>: Send + Sync {
    /// The stage on the device.
    fn stage(&self, spec: StageSpec<F, EF>) -> Box<dyn DeviceStage<EF>>;
}

/// A stage on the device, driven round by round.
pub trait DeviceStage<EF>: Send + Sync {
    /// The rows left.
    fn num_evals(&self) -> usize;

    /// For each `(node, value)` in order, the constraint sum of every table and the
    /// interaction sum of every group over the pairs of rows, weighted by the equality at
    /// [`StageSpec::eq_point`] as the folds so far left it, with a slot's constraints
    /// evaluated at node zero only if `include_constraint_node_zero`.
    /// The boundary selectors are `first` at row zero and `last` at the last row of the
    /// current cube, the transition selector one less `last` there.
    fn round(
        &self,
        nodes: &[(usize, EF)],
        include_constraint_node_zero: bool,
        first: EF,
        last: EF,
    ) -> Vec<(Vec<EF>, Vec<EF>)>;

    /// Bind the current variable to `r`, keeping the wrapped next-row values first.
    fn fold(&mut self, r: EF);

    /// The wrapped next-row value of every column after the last fold.
    fn next_tail(&self) -> Vec<EF>;

    /// Every column's single value once the rows are folded away.
    fn openings(&self) -> Vec<EF>;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum LeafKey<F> {
    Local(u32),
    Next(u32),
    Const(F),
    First,
    Last,
    Transition,
}

/// A program under construction: single-assignment values first, registers at the end.
struct Compiler<F> {
    code: Vec<Instruction>,
    produces: Vec<bool>,
    constants: Vec<F>,
    constant_index: HashMap<F, u32>,
    leaves: HashMap<LeafKey<F>, u32>,
    nodes: HashMap<*const SymbolicExpression<F>, u32>,
    failed: bool,
}

impl<F: Field> Compiler<F> {
    fn new() -> Self {
        Self {
            code: Vec::new(),
            produces: Vec::new(),
            constants: Vec::new(),
            constant_index: HashMap::new(),
            leaves: HashMap::new(),
            nodes: HashMap::new(),
            failed: false,
        }
    }

    fn value(&mut self, op: u32, a: u32, b: u32) -> u32 {
        let id = self.code.len() as u32;
        self.code.push(Instruction { op, a, b, dst: id });
        self.produces.push(true);
        id
    }

    fn effect(&mut self, op: u32, a: u32, b: u32, dst: u32) {
        self.code.push(Instruction { op, a, b, dst });
        self.produces.push(false);
    }

    fn leaf(&mut self, key: LeafKey<F>) -> u32 {
        if let Some(&id) = self.leaves.get(&key) {
            return id;
        }
        let id = match key {
            LeafKey::Local(column) => self.value(OP_LOCAL, column, 0),
            LeafKey::Next(column) => self.value(OP_NEXT, column, 0),
            LeafKey::Const(value) => {
                let index = *self.constant_index.entry(value).or_insert_with(|| {
                    self.constants.push(value);
                    (self.constants.len() - 1) as u32
                });
                self.value(OP_CONST, index, 0)
            }
            LeafKey::First => self.value(OP_FIRST, 0, 0),
            LeafKey::Last => self.value(OP_LAST, 0, 0),
            LeafKey::Transition => self.value(OP_TRANSITION, 0, 0),
        };
        self.leaves.insert(key, id);
        id
    }

    fn constant(&mut self, value: F) -> u32 {
        self.leaf(LeafKey::Const(value))
    }

    /// Start a top-level expression: values are shared within one expression and reloaded
    /// or recomputed across them, which bounds the registers a program holds live.
    fn scope(&mut self) {
        self.leaves.clear();
        self.nodes.clear();
    }

    fn expr<A>(&mut self, expr: &SymbolicExpression<F>, slot: &AirSlot<'_, A>, public: &[F]) -> u32 {
        let key: *const SymbolicExpression<F> = expr;
        if let Some(&id) = self.nodes.get(&key) {
            return id;
        }
        let id = match expr {
            SymbolicExpr::Leaf(leaf) => match leaf {
                BaseLeaf::Variable(variable) => {
                    let (base, offset) = match variable.entry {
                        BaseEntry::Main { offset } => (slot.main_offset, offset),
                        BaseEntry::Preprocessed { offset } => (slot.preprocessed_offset, offset),
                        BaseEntry::Periodic => (slot.periodic_offset, 0),
                        BaseEntry::Public => {
                            let value = public[variable.index];
                            let id = self.constant(value);
                            self.nodes.insert(key, id);
                            return id;
                        }
                    };
                    let column = (base + variable.index) as u32;
                    match offset {
                        0 => self.leaf(LeafKey::Local(column)),
                        1 => self.leaf(LeafKey::Next(column)),
                        _ => {
                            self.failed = true;
                            self.leaf(LeafKey::Local(column))
                        }
                    }
                }
                BaseLeaf::IsFirstRow => self.leaf(LeafKey::First),
                BaseLeaf::IsLastRow => self.leaf(LeafKey::Last),
                BaseLeaf::IsTransition => self.leaf(LeafKey::Transition),
                BaseLeaf::Constant(value) => self.constant(*value),
            },
            SymbolicExpr::Add { x, y, .. } => {
                let (a, b) = (self.expr(x, slot, public), self.expr(y, slot, public));
                self.value(OP_ADD, a, b)
            }
            SymbolicExpr::Sub { x, y, .. } => {
                let (a, b) = (self.expr(x, slot, public), self.expr(y, slot, public));
                self.value(OP_SUB, a, b)
            }
            SymbolicExpr::Neg { x, .. } => {
                let a = self.expr(x, slot, public);
                self.value(OP_NEG, a, 0)
            }
            SymbolicExpr::Mul { x, y, .. } => {
                let (a, b) = (self.expr(x, slot, public), self.expr(y, slot, public));
                self.value(OP_MUL, a, b)
            }
        };
        self.nodes.insert(key, id);
        id
    }

    /// Allocate registers: a value's register is free after its last use.
    fn finish(self) -> Option<Program<F>> {
        if self.failed {
            return None;
        }
        let n = self.code.len();
        let mut last_use: Vec<usize> = vec![0usize; n];
        for (i, instruction) in self.code.iter().enumerate() {
            for operand in Self::operands(instruction) {
                last_use[operand as usize] = i;
            }
        }
        let mut register: Vec<u32> = vec![u32::MAX; n];
        let mut free: Vec<u32> = Vec::new();
        let mut registers = 0u32;
        let mut code = Vec::with_capacity(n);
        for (i, instruction) in self.code.iter().enumerate() {
            let mut out = *instruction;
            let operands = Self::operands(instruction);
            for (k, operand) in operands.iter().enumerate() {
                let reg = register[*operand as usize];
                debug_assert_ne!(reg, u32::MAX, "an operand is defined before its use");
                if k == 0 {
                    out.a = reg;
                } else {
                    out.b = reg;
                }
            }
            for operand in &operands {
                if last_use[*operand as usize] == i {
                    free.push(register[*operand as usize]);
                }
            }
            if self.produces[i] {
                let reg = free.pop().unwrap_or_else(|| {
                    registers += 1;
                    registers - 1
                });
                register[i] = reg;
                out.dst = reg;
                if last_use[i] == 0 && !Self::used_later(&self.code, i) {
                    free.push(reg);
                }
            }
            code.push(out);
        }
        Some(Program { code, registers: registers as usize, constants: self.constants })
    }

    fn operands(instruction: &Instruction) -> Vec<u32> {
        match instruction.op {
            OP_ADD | OP_SUB | OP_MUL => alloc::vec![instruction.a, instruction.b],
            OP_NEG | OP_CONSTRAINT | OP_FIELD | OP_INTERACTION => alloc::vec![instruction.a],
            _ => Vec::new(),
        }
    }

    fn used_later(code: &[Instruction], id: usize) -> bool {
        code[id + 1..]
            .iter()
            .any(|instruction| Self::operands(instruction).contains(&(id as u32)))
    }
}

/// The program of `slot`'s AIR, or `None` if the AIR has what the device cannot run.
fn compile_slot<F, EF, A>(slot: &AirSlot<'_, A>, public: &[F]) -> Option<Program<F>>
where
    F: Field,
    EF: ExtensionField<F>,
    A: BaseAir<F> + Air<InteractionSymbolicBuilder<F, EF>>,
{
    let layout = AirLayout::from_air::<F>(slot.air);
    let builder = InteractionSymbolicBuilder::<F, EF>::from_air(slot.air, layout);
    if !builder.extension_constraints().is_empty() || !builder.exclusive_interactions().is_empty() {
        return None;
    }
    let mut compiler = Compiler::<F>::new();
    let constraints = builder.base_constraints();
    for (index, constraint) in constraints.iter().enumerate() {
        compiler.scope();
        let value = compiler.expr(constraint, slot, public);
        compiler.effect(OP_CONSTRAINT, value, index as u32, 0);
    }
    let mut index = constraints.len();
    for cell in slot.air.public_boundary_io() {
        compiler.scope();
        let value = compiler.leaf(LeafKey::Local((slot.main_offset + cell.column) as u32));
        let expected = compiler.constant(public[cell.public_value]);
        let difference = compiler.value(OP_SUB, value, expected);
        let selector = compiler.leaf(match cell.end {
            BoundaryEnd::First => LeafKey::First,
            BoundaryEnd::Last => LeafKey::Last,
        });
        let product = compiler.value(OP_MUL, selector, difference);
        compiler.effect(OP_CONSTRAINT, product, index as u32, 0);
        index += 1;
    }
    let num_local = builder.local_interactions().len();
    for (g, interaction) in builder.global_interactions().iter().enumerate() {
        for (k, field) in interaction.fields.iter().enumerate() {
            compiler.scope();
            let value = compiler.expr(field, slot, public);
            compiler.effect(OP_FIELD, value, k as u32, 0);
        }
        compiler.scope();
        let count = compiler.expr(&interaction.count, slot, public);
        compiler.effect(OP_INTERACTION, count, (num_local + g) as u32, 0);
    }
    for (l, interaction) in builder.local_interactions().iter().enumerate() {
        for (t, (fields, count)) in interaction.tuples.iter().enumerate() {
            for (k, field) in fields.iter().enumerate() {
                compiler.scope();
                let value = compiler.expr(field, slot, public);
                compiler.effect(OP_FIELD, value, k as u32, 0);
            }
            let (count, _) = count.clone().into_parts();
            let count = compiler.expr(&count, slot, public);
            compiler.effect(OP_INTERACTION, count, l as u32, t as u32);
        }
    }
    compiler.finish()
}

impl<'air, 'data, A, F, EF> RoundStateBase<'air, 'data, A, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
    A: BaseAir<F>,
{
    /// The stage for a device.
    ///
    /// # Panics
    /// Panics if the stage has what the device cannot run: a periodic column, or a constraint
    /// the slot compiler does not take. There is no host path once a device is given.
    fn device_spec(&self) -> StageSpec<F, EF>
    where
        A: Air<InteractionSymbolicBuilder<F, EF>>,
    {
        let width = self.total_width();
        assert!(
            self.periodic.iter().all(Option::is_none),
            "a periodic column in a device zerocheck stage: periodic columns are not on the device"
        );
        let mut tables = alloc::vec![
            TableSpec {
                main_offset: 0,
                main_width: 0,
                preprocessed: None
            };
            self.tables.len()
        ];
        for slot in &self.slots {
            tables[slot.stage_index] = TableSpec {
                main_offset: slot.main_offset,
                main_width: slot.main_width,
                preprocessed: self.preprocessed[slot.stage_index]
                    .map(|_| (slot.preprocessed_offset, slot.preprocessed_width)),
            };
        }
        let next_columns = next_row_runs(&self.slots)
            .into_iter()
            .flat_map(|run| run.map(|c| c as u32))
            .collect();
        let slots = self
            .slots
            .iter()
            .map(|slot| {
                let program = compile_slot::<F, EF, A>(slot, self.public_values[slot.stage_index])
                    .unwrap_or_else(|| {
                        panic!(
                            "the device zerocheck cannot compile the constraints of stage table {}",
                            slot.stage_index
                        )
                    });
                SlotSpec {
                    stage: slot.stage_index,
                    constraint_degree: slot.constraint_degree,
                    interaction: slot.interaction.map(|interaction| InteractionSpec {
                        degree: interaction.degree,
                        group: interaction.group_index,
                        link: interaction.link_index,
                    }),
                    alpha_powers: self.alpha_powers[slot.stage_index].clone(),
                    program,
                }
            })
            .collect::<Vec<_>>();
        let links = self
            .coupling
            .links
            .iter()
            .map(|link| LinkSpec {
                num_local: link.num_local_lookups,
                lookups: link
                    .lookups
                    .iter()
                    .map(|lookup| LookupSpec {
                        offset: lookup.theta_bus_offset,
                        weights: lookup.block_weights.clone(),
                    })
                    .collect(),
            })
            .collect();
        StageSpec {
            num_vars: self.tables[0].num_variables(),
            width,
            instances: self.instances.clone(),
            tables,
            next_columns,
            slots,
            theta_beta_powers: self.coupling.theta_beta_powers.clone(),
            links,
            stages: self.tables.len(),
            groups: self.interaction_groups.len(),
            eq_point: self.tau.as_slice()[1..].to_vec(),
        }
    }

    /// The first round on the device, which takes the stage first.
    pub(crate) fn device_round0(&mut self, device: &dyn ZerocheckDevice<F, EF>) -> Vec<EF>
    where
        A: Air<InteractionSymbolicBuilder<F, EF>>,
    {
        if self.device.is_none() {
            let spec = tracing::info_span!("device stage spec").in_scope(|| self.device_spec());
            self.device = Some(tracing::info_span!("device stage").in_scope(|| device.stage(spec)));
        }
        let nodes = node_values::<EF>(evaluated_nodes(&self.slots, self.degree(), false));
        let sums = self
            .device
            .as_ref()
            .expect("the stage is on the device")
            .round(&nodes, false, EF::ONE, EF::ONE);
        let (constraint_evals, interaction_evals) = self.collect_evals(&nodes, sums, false);
        finish_round(
            &mut self.constraint_groups,
            &mut self.interaction_groups,
            &self.betas,
            self.eta,
            &constraint_evals,
            &interaction_evals,
            self.tau.as_slice()[0],
        )
    }

    fn collect_evals(
        &self,
        nodes: &[(usize, EF)],
        sums: Vec<(Vec<EF>, Vec<EF>)>,
        include_constraint_node_zero: bool,
    ) -> (Vec<Vec<EF>>, Vec<Vec<EF>>) {
        collect_evals(
            &self.slots,
            self.tables.len(),
            self.interaction_groups.iter().map(|group| group.degree).collect(),
            nodes,
            sums,
            include_constraint_node_zero,
        )
    }

    /// Fold the first variable on the device.
    pub(crate) fn fold0_device(mut self, r: EF) -> RoundStateExt<'air, 'data, A, F, EF> {
        self.fold_claims(r);
        let mut device = self.device.take().expect("the stage is on the device");
        device.fold(r);
        let next_tail = device.next_tail();
        RoundStateExt {
            public_values: self.public_values,
            alpha: self.alpha,
            alpha_powers: self.alpha_powers,
            betas: self.betas,
            constraint_groups: self.constraint_groups,
            interaction_groups: self.interaction_groups,
            slots: self.slots,
            tau: self.tau,
            round: 1,
            columns: ExtColumns::Scalar(Vec::new()),
            next_tail,
            coupling: self.coupling,
            lookup_scale: self.eta,
            boundary: BoundaryEvals::new(EF::ONE - r, r, EF::ONE - r),
            device: Some(device),
        }
    }
}

impl<'air, 'data, A, F, EF> RoundStateExt<'air, 'data, A, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
    A: BaseAir<F>,
{
    /// A later round on the device.
    pub(crate) fn device_round(&mut self) -> Vec<EF> {
        let nodes = node_values::<EF>(evaluated_nodes(&self.slots, self.degree(), true));
        let device = self.device.as_ref().expect("the stage is on the device");
        let sums = device.round(
            &nodes,
            true,
            self.boundary.first,
            self.boundary.last,
        );
        let (constraint_evals, interaction_evals) = collect_evals(
            &self.slots,
            self.alpha_powers.len(),
            self.interaction_groups.iter().map(|group| group.degree).collect(),
            &nodes,
            sums,
            true,
        );
        finish_round(
            &mut self.constraint_groups,
            &mut self.interaction_groups,
            &self.betas,
            self.lookup_scale,
            &constraint_evals,
            &interaction_evals,
            self.tau.as_slice()[self.round],
        )
    }

    /// Fold the current variable on the device.
    pub(crate) fn fold_device(&mut self, r: EF) {
        self.fold_claims(r);
        let device = self.device.as_mut().expect("the stage is on the device");
        device.fold(r);
        self.next_tail = device.next_tail();
        self.boundary.apply(r);
        self.round += 1;
    }

    /// The openings from the device's folded columns.
    pub(crate) fn device_openings(self) -> Vec<(usize, AirOpenings<EF>)> {
        let device = self.device.as_ref().expect("the stage is on the device");
        let local: Vec<EF> = device.openings();
        let all_next = self.next_tail;
        self.slots
            .into_iter()
            .map(|slot| {
                let main_end = slot.main_offset + slot.main_width;
                let preprocessed_end = slot.preprocessed_offset + slot.preprocessed_width;
                let next = slot
                    .air
                    .main_next_row_columns()
                    .into_iter()
                    .map(|column| all_next[slot.main_offset + column])
                    .collect();
                let preprocessed_next = slot
                    .air
                    .preprocessed_next_row_columns()
                    .into_iter()
                    .map(|column| all_next[slot.preprocessed_offset + column])
                    .collect();
                (
                    slot.caller_index,
                    AirOpenings {
                        local: local[slot.main_offset..main_end].to_vec(),
                        next,
                        preprocessed_local: local[slot.preprocessed_offset..preprocessed_end]
                            .to_vec(),
                        preprocessed_next,
                    },
                )
            })
            .collect()
    }
}

fn node_values<EF: Field>(nodes: impl Iterator<Item = usize>) -> Vec<(usize, EF)> {
    nodes.map(|node| (node, EF::interpolation_node(node))).collect()
}

/// The device's per-node sums laid out as the host's scratch: one vector per table of
/// `constraint_degree` entries and one per interaction group of its degree, entry
/// `node - 1` for node `node` and entry `0` for node zero.
fn collect_evals<A, EF: Field>(
    slots: &[AirSlot<'_, A>],
    tables: usize,
    group_degrees: Vec<usize>,
    nodes: &[(usize, EF)],
    sums: Vec<(Vec<EF>, Vec<EF>)>,
    include_constraint_node_zero: bool,
) -> (Vec<Vec<EF>>, Vec<Vec<EF>>) {
    let mut constraint_evals: Vec<Vec<EF>> = alloc::vec![Vec::new(); tables];
    for slot in slots {
        constraint_evals[slot.stage_index] = EF::zero_vec(slot.constraint_degree);
    }
    let mut interaction_evals: Vec<Vec<EF>> =
        group_degrees.iter().map(|&degree| EF::zero_vec(degree)).collect();
    for (&(node, _), (constraints, interactions)) in nodes.iter().zip(sums) {
        let index = if node == 0 { 0 } else { node - 1 };
        for slot in slots {
            let enabled = slot.enabled_families(node, include_constraint_node_zero);
            if enabled.constraints {
                constraint_evals[slot.stage_index][index] += constraints[slot.stage_index];
            }
        }
        for (group, evals) in interaction_evals.iter_mut().enumerate() {
            if index < evals.len() {
                evals[index] += interactions[group];
            }
        }
    }
    (constraint_evals, interaction_evals)
}
