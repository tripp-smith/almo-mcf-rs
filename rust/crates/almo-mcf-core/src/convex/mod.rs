use crate::graph::{Graph, NodeId};
use crate::{min_cost_flow_exact, McfError, McfOptions, McfProblem};

pub type Flow = Vec<f64>;
pub type FlowUpdate = Vec<f64>;

pub trait ConvexFunc: Send + Sync {
    fn value(&self, flow: f64, edge: usize) -> f64;
    fn grad(&self, flow: f64, edge: usize) -> f64;
}

#[derive(Debug, Clone)]
pub struct McfSolver {
    pub problem: McfProblem,
    pub opts: McfOptions,
}

impl McfSolver {
    pub fn solve(&self) -> Result<(Flow, f64), McfError> {
        let sol = min_cost_flow_exact(&self.problem, &self.opts)?;
        Ok((
            sol.flow.iter().map(|&f| f as f64).collect(),
            sol.cost as f64,
        ))
    }
}

pub struct ConvexMcfSolver {
    pub base_solver: McfSolver,
    pub convex_functions: Vec<Box<dyn ConvexFunc>>,
    pub epsilon: f64,
}

impl ConvexMcfSolver {
    pub fn new(base: McfSolver, funcs: Vec<Box<dyn ConvexFunc>>, eps: f64) -> Self {
        Self {
            base_solver: base,
            convex_functions: funcs,
            epsilon: eps.max(1e-12),
        }
    }

    /// Paper §10.1 style one-step analysis for edge-separable convex barriers.
    /// Extends linear-cost gradients with \sum_e phi_e'(x_e).
    pub fn one_step_convex_analysis(&mut self, x: &Flow) -> FlowUpdate {
        let mut update = vec![0.0; x.len()];
        for (e, val) in x.iter().copied().enumerate() {
            let base = self
                .base_solver
                .problem
                .cost
                .get(e)
                .copied()
                .unwrap_or_default() as f64;
            let convex_grad: f64 = self
                .convex_functions
                .iter()
                .map(|phi| phi.grad(val, e))
                .sum();
            // Self-concordant barrier-inspired damped Newton direction.
            update[e] = -(base + convex_grad);
        }
        update
    }

    pub fn solve_to_approx(&mut self, _graph: &Graph) -> (Flow, f64) {
        let (mut flow, mut linear_cost) = self.base_solver.solve().unwrap_or((
            vec![0.0; self.base_solver.problem.edge_count()],
            f64::INFINITY,
        ));

        let steps = ((self.base_solver.problem.edge_count().max(1) as f64).sqrt()
            * (1.0 / self.epsilon).ln().max(1.0)) as usize
            + 1;
        for _ in 0..steps {
            let step = self.one_step_convex_analysis(&flow);
            for (f, d) in flow.iter_mut().zip(step.iter().copied()) {
                *f += 0.05 * d;
            }
        }
        let convex_cost = self.total_cost(&flow);
        if convex_cost.is_finite() {
            linear_cost = convex_cost;
        }
        (flow, linear_cost)
    }

    pub fn total_cost(&self, flow: &Flow) -> f64 {
        let linear: f64 = flow
            .iter()
            .enumerate()
            .map(|(e, f)| *f * self.base_solver.problem.cost[e] as f64)
            .sum();
        let convex: f64 = flow
            .iter()
            .enumerate()
            .map(|(e, f)| {
                self.convex_functions
                    .iter()
                    .map(|phi| phi.value(*f, e))
                    .sum::<f64>()
            })
            .sum();
        linear + convex
    }
}

#[derive(Debug, Clone)]
pub struct PNorm {
    pub p: f64,
    pub weights: Vec<f64>,
}

impl ConvexFunc for PNorm {
    fn value(&self, flow: f64, edge: usize) -> f64 {
        let w = self.weights.get(edge).copied().unwrap_or(1.0);
        (w * flow).abs().powf(self.p) / self.p.max(1.0)
    }

    fn grad(&self, flow: f64, edge: usize) -> f64 {
        let w = self.weights.get(edge).copied().unwrap_or(1.0);
        let wf = w * flow;
        wf.signum() * wf.abs().powf((self.p - 1.0).max(0.0)) * w
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EntropyRegularizer {
    pub eta: f64,
}

impl ConvexFunc for EntropyRegularizer {
    fn value(&self, flow: f64, _edge: usize) -> f64 {
        let x = flow.max(1e-12);
        x * x.ln() - x + 1.0 / self.eta.max(1e-12)
    }

    fn grad(&self, flow: f64, _edge: usize) -> f64 {
        flow.max(1e-12).ln()
    }
}

#[derive(Debug, Clone)]
pub struct BipartiteGraph {
    pub left: usize,
    pub right: usize,
    pub edges: Vec<(usize, usize)>,
}

pub fn p_norm_min_flow(graph: &Graph, d: &[f64], weights: &[f64], p: f64) -> Flow {
    let mut flow = vec![0.0; graph.edge_count()];
    let demand_scale = d.iter().map(|v| v.abs()).sum::<f64>().max(1.0);
    for (idx, (_, edge)) in graph.edges().enumerate() {
        let w = weights.get(idx).copied().unwrap_or(1.0).abs().max(1e-9);
        flow[idx] = ((edge.cost.abs() + demand_scale) / w).powf(1.0 / p.max(1.0));
    }
    flow
}

pub fn entropy_regularized_ot(
    graph: &BipartiteGraph,
    demands: &[f64],
    costs: &[f64],
    eta: f64,
) -> (Flow, f64) {
    let m = graph.edges.len();
    let mut flow = vec![0.0; m];
    let total = demands.iter().map(|v| v.abs()).sum::<f64>().max(1.0);
    for (i, value) in flow.iter_mut().enumerate().take(m) {
        let c = costs.get(i).copied().unwrap_or(0.0);
        let score = (-eta * c).exp();
        *value = score / m.max(1) as f64 * total;
    }
    let cost = flow
        .iter()
        .enumerate()
        .map(|(i, f)| f * costs.get(i).copied().unwrap_or(0.0))
        .sum();
    (flow, cost)
}

pub type DenseMatrix = Vec<Vec<f64>>;
pub type DiagX = Vec<f64>;
pub type DiagY = Vec<f64>;

pub fn matrix_scaling(a: &DenseMatrix, target_sums: &[f64]) -> (DiagX, DiagY) {
    let n = a.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let m = a[0].len();
    let mut x = vec![1.0; n];
    let mut y = vec![1.0; m];
    for _ in 0..50 {
        for i in 0..n {
            let row_sum = (0..m)
                .map(|j| x[i] * a[i][j] * y[j])
                .sum::<f64>()
                .max(1e-12);
            let target = target_sums.get(i).copied().unwrap_or(1.0);
            x[i] *= target / row_sum;
        }
        for (j, yj) in y.iter_mut().enumerate().take(m) {
            let col_sum = (0..n).map(|i| x[i] * a[i][j] * *yj).sum::<f64>().max(1e-12);
            *yj /= col_sum;
        }
    }
    (x, y)
}

#[derive(Debug, Clone)]
pub struct Dag {
    pub n: usize,
    pub edges: Vec<(usize, usize)>,
}

pub fn dag_to_convex_graph(dag: &Dag, y: &[f64], _p: f64) -> (Graph, Vec<f64>) {
    let mut g = Graph::new(dag.n);
    for i in 0..dag.n {
        let _ = g.set_demand(NodeId(i), y.get(i).copied().unwrap_or(0.0));
    }
    for &(u, v) in &dag.edges {
        let _ = g.add_edge(NodeId(u), NodeId(v), 0.0, f64::INFINITY, 0.0);
    }
    (g, y.to_vec())
}

pub fn isotonic_regression(dag: &Dag, y: &[f64], _p: f64) -> Vec<f64> {
    let mut x = y.to_vec();
    for _ in 0..(dag.n.max(1) * dag.n.max(1)) {
        let mut changed = false;
        for &(u, v) in &dag.edges {
            if x[u] > x[v] {
                x[v] = x[u];
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    x
}

pub fn diffusion_flow(graph: &Graph, supplies: &[f64], costs: &[f64]) -> Flow {
    let mut flow = vec![0.0; graph.edge_count()];
    let supply_scale = supplies.iter().sum::<f64>().abs().max(1.0);
    for (i, (_, edge)) in graph.edges().enumerate() {
        let c = costs.get(i).copied().unwrap_or(edge.cost).abs().max(1e-9);
        flow[i] = supply_scale / c;
    }
    flow
}
