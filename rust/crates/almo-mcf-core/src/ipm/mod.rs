use crate::graph::min_cost_flow::MinCostFlow;
use crate::min_ratio::dynamic::FullDynamicOracle;
use crate::min_ratio::{MinRatioOracle, OracleQuery};
use crate::numerics::duality_gap_proxy;
use crate::{McfError, McfOptions, McfProblem, Strategy};
use std::time::Instant;

mod potential;
mod search;

pub use potential::Potential;

#[derive(Debug, Default, Clone)]
pub struct IpmState {
    pub flow: Vec<f64>,
    pub potential: f64,
}

#[derive(Debug, Default, Clone)]
pub struct IpmStats {
    pub iterations: usize,
    pub last_step_size: f64,
    pub potentials: Vec<f64>,
    pub last_gap: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpmTermination {
    Converged,
    IterationLimit,
    TimeLimit,
    NoImprovingCycle,
}

#[derive(Debug, Clone)]
pub struct IpmResult {
    pub flow: Vec<f64>,
    pub stats: IpmStats,
    pub termination: IpmTermination,
}

#[derive(Debug, Clone)]
pub struct FeasibleFlow {
    pub flow: Vec<f64>,
}

pub fn initialize_feasible_flow(problem: &McfProblem) -> Result<FeasibleFlow, McfError> {
    let base_flow = solve_feasible_flow(
        problem.node_count,
        &problem.tails,
        &problem.heads,
        &problem.lower,
        &problem.upper,
        &problem.demands,
    )?;
    let flow = push_inside_strict(problem, &base_flow)?;
    Ok(FeasibleFlow { flow })
}

pub fn run_ipm(problem: &McfProblem, opts: &McfOptions) -> Result<IpmResult, McfError> {
    let feasible = initialize_feasible_flow(problem)?;
    let mut flow = feasible.flow;
    let lower: Vec<f64> = problem.lower.iter().map(|&v| v as f64).collect();
    let upper: Vec<f64> = problem.upper.iter().map(|&v| v as f64).collect();
    let cost: Vec<f64> = problem.cost.iter().map(|&v| v as f64).collect();
    let potential = Potential::new(&upper);

    let mut fallback_oracle = None;
    let mut dynamic_oracle = None;
    match opts.strategy {
        Strategy::PeriodicRebuild { rebuild_every } => {
            fallback_oracle = Some(MinRatioOracle::new(opts.seed, rebuild_every));
        }
        Strategy::FullDynamic => {
            dynamic_oracle = Some(FullDynamicOracle::new(opts.seed, 3, 1, 10, 0.1));
        }
    }

    let start = Instant::now();
    let mut stats = IpmStats {
        iterations: 0,
        last_step_size: 0.0,
        potentials: Vec::new(),
        last_gap: f64::INFINITY,
    };
    let mut termination = IpmTermination::IterationLimit;

    for iter in 0..opts.max_iters {
        if let Some(limit) = opts.time_limit_ms {
            if start.elapsed().as_millis() as u64 >= limit {
                termination = IpmTermination::TimeLimit;
                break;
            }
        }

        let (gradient, lengths) =
            compute_gradient_and_lengths(&potential, &cost, &flow, &lower, &upper);
        let current_potential = potential.value(&cost, &flow, &lower, &upper);
        stats.potentials.push(current_potential);
        stats.last_gap = duality_gap_proxy(&gradient, &flow);
        if stats.last_gap < opts.tolerance {
            termination = IpmTermination::Converged;
            stats.iterations = iter;
            break;
        }

        let best = if let Some(oracle) = fallback_oracle.as_mut() {
            oracle
                .best_cycle(OracleQuery {
                    iter,
                    node_count: problem.node_count,
                    tails: &problem.tails,
                    heads: &problem.heads,
                    gradients: &gradient,
                    lengths: &lengths,
                })
                .map_err(|err| McfError::InvalidInput(format!("{err:?}")))?
        } else if let Some(oracle) = dynamic_oracle.as_mut() {
            oracle
                .best_cycle(
                    iter,
                    problem.node_count,
                    &problem.tails,
                    &problem.heads,
                    &gradient,
                    &lengths,
                )
                .map_err(|err| McfError::InvalidInput(format!("{err:?}")))?
        } else {
            None
        };
        let Some(best) = best else {
            termination = IpmTermination::NoImprovingCycle;
            stats.iterations = iter;
            break;
        };

        if best.ratio >= -opts.tolerance {
            termination = IpmTermination::Converged;
            stats.iterations = iter;
            break;
        }

        let mut delta = vec![0.0_f64; flow.len()];
        for (edge_id, dir) in best.cycle_edges {
            delta[edge_id] += dir as f64;
        }

        if let Some((candidate_flow, step)) = search::line_search(
            &flow,
            &delta,
            &cost,
            &lower,
            &upper,
            &potential,
            current_potential,
        ) {
            flow = candidate_flow;
            stats.last_step_size = step;
        } else {
            termination = IpmTermination::NoImprovingCycle;
            stats.iterations = iter;
            break;
        }

        stats.iterations = iter + 1;
    }

    Ok(IpmResult {
        flow,
        stats,
        termination,
    })
}

fn compute_gradient_and_lengths(
    potential: &Potential,
    cost: &[f64],
    flow: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let gradient = potential.gradient(cost, flow, lower, upper);
    let lengths = potential.lengths(flow, lower, upper);
    (gradient, lengths)
}

fn solve_feasible_flow(
    node_count: usize,
    tails: &[u32],
    heads: &[u32],
    lower: &[i64],
    upper: &[i64],
    demands: &[i64],
) -> Result<Vec<i64>, McfError> {
    let n = node_count;
    let m = lower.len();
    if tails.len() != m || heads.len() != m || upper.len() != m {
        return Err(McfError::InvalidInput(
            "edge arrays must have identical length".to_string(),
        ));
    }
    if demands.len() != n {
        return Err(McfError::InvalidInput(
            "demand array length mismatch".to_string(),
        ));
    }

    let mut demand = demands.to_vec();
    let mut residual_upper = vec![0_i64; m];
    for (i, residual) in residual_upper.iter_mut().enumerate() {
        let lo = lower[i];
        let up = upper[i];
        if lo > up {
            return Err(McfError::InvalidInput(
                "lower bound exceeds upper bound".to_string(),
            ));
        }
        *residual = up
            .checked_sub(lo)
            .ok_or_else(|| McfError::InvalidInput("capacity overflow".to_string()))?;
        let tail = tails[i] as usize;
        let head = heads[i] as usize;
        demand[tail] = demand[tail]
            .checked_add(lo)
            .ok_or_else(|| McfError::InvalidInput("demand overflow".to_string()))?;
        demand[head] = demand[head]
            .checked_sub(lo)
            .ok_or_else(|| McfError::InvalidInput("demand overflow".to_string()))?;
    }

    let total_nodes = n + 2;
    let source = n;
    let sink = n + 1;
    let mut mcf = MinCostFlow::new(total_nodes);
    let mut edge_refs = Vec::with_capacity(m);

    for (i, &cap) in residual_upper.iter().enumerate() {
        if cap < 0 {
            return Err(McfError::InvalidInput(
                "negative residual capacity".to_string(),
            ));
        }
        let tail = tails[i] as usize;
        let head = heads[i] as usize;
        let (idx, _rev) = mcf.add_edge(tail, head, cap, 0);
        edge_refs.push((tail, idx, cap));
    }

    let mut total_demand = 0_i64;
    for (node, &b) in demand.iter().enumerate() {
        if b > 0 {
            mcf.add_edge(node, sink, b, 0);
        } else if b < 0 {
            mcf.add_edge(source, node, -b, 0);
            total_demand += -b;
        }
    }

    if total_demand > 0 {
        mcf.min_cost_flow(source, sink, total_demand)?;
    }

    let mut flow = vec![0_i64; m];
    for (i, (tail, idx, cap)) in edge_refs.iter().enumerate() {
        let edge = &mcf.graph[*tail][*idx];
        let used = cap - edge.cap;
        flow[i] = lower[i]
            .checked_add(used)
            .ok_or_else(|| McfError::InvalidInput("flow overflow".to_string()))?;
    }

    Ok(flow)
}

fn push_inside_strict(problem: &McfProblem, base_flow: &[i64]) -> Result<Vec<f64>, McfError> {
    const SCALE: i64 = 2;
    const EPS_SCALED: i64 = 1;
    let m = problem.edge_count();
    let n = problem.node_count;
    if base_flow.len() != m {
        return Err(McfError::InvalidInput(
            "base flow length mismatch".to_string(),
        ));
    }

    let mut lower_delta = vec![0_i64; m];
    let mut upper_delta = vec![0_i64; m];
    for i in 0..m {
        let lo_scaled = scale_checked(problem.lower[i], SCALE)?
            .checked_add(EPS_SCALED)
            .ok_or_else(|| McfError::InvalidInput("scaled lower overflow".to_string()))?;
        let up_scaled = scale_checked(problem.upper[i], SCALE)?
            .checked_sub(EPS_SCALED)
            .ok_or_else(|| McfError::InvalidInput("scaled upper overflow".to_string()))?;
        if lo_scaled > up_scaled {
            return Err(McfError::Infeasible);
        }
        let base_scaled = scale_checked(base_flow[i], SCALE)?;
        lower_delta[i] = lo_scaled
            .checked_sub(base_scaled)
            .ok_or_else(|| McfError::InvalidInput("delta lower overflow".to_string()))?;
        upper_delta[i] = up_scaled
            .checked_sub(base_scaled)
            .ok_or_else(|| McfError::InvalidInput("delta upper overflow".to_string()))?;
    }

    let zero_demands = vec![0_i64; n];
    let delta = solve_feasible_flow(
        n,
        &problem.tails,
        &problem.heads,
        &lower_delta,
        &upper_delta,
        &zero_demands,
    )?;
    let mut flow = vec![0_f64; m];
    for i in 0..m {
        let total_scaled = scale_checked(base_flow[i], SCALE)?
            .checked_add(delta[i])
            .ok_or_else(|| McfError::InvalidInput("flow overflow".to_string()))?;
        flow[i] = total_scaled as f64 / SCALE as f64;
    }
    Ok(flow)
}

fn scale_checked(value: i64, scale: i64) -> Result<i64, McfError> {
    value
        .checked_mul(scale)
        .ok_or_else(|| McfError::InvalidInput("scaled value overflow".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_feasible(problem: &McfProblem, flow: &[f64]) {
        assert_eq!(flow.len(), problem.edge_count());
        let mut balance = vec![0_f64; problem.node_count];
        for (idx, (&f, (&tail, &head))) in flow
            .iter()
            .zip(problem.tails.iter().zip(problem.heads.iter()))
            .enumerate()
        {
            let lo = problem.lower[idx] as f64;
            let up = problem.upper[idx] as f64;
            assert!(
                f >= lo - 1e-9 && f <= up + 1e-9,
                "flow out of bounds at edge {idx}: {f} not in [{lo}, {up}]"
            );
            balance[tail as usize] -= f;
            balance[head as usize] += f;
        }
        for (node, (&b, &d)) in balance.iter().zip(problem.demands.iter()).enumerate() {
            assert!(
                (b - d as f64).abs() <= 1e-9,
                "node {node} imbalance mismatch: got {b}, expected {d}"
            );
        }
    }

    #[test]
    fn initializes_feasible_flow_with_lower_bounds() {
        let problem = McfProblem::new(
            vec![0, 1],
            vec![1, 2],
            vec![1, 1],
            vec![3, 3],
            vec![1, 1],
            vec![-2, 0, 2],
        )
        .unwrap();
        let feasible = initialize_feasible_flow(&problem).unwrap();
        assert_feasible(&problem, &feasible.flow);
    }

    #[test]
    fn initializer_rejects_infeasible_instance() {
        let problem =
            McfProblem::new(vec![0], vec![1], vec![0], vec![1], vec![1], vec![-2, 2]).unwrap();
        let err = initialize_feasible_flow(&problem).unwrap_err();
        assert!(matches!(err, McfError::Infeasible));
    }

    #[test]
    fn initializer_rejects_missing_strict_interior() {
        let problem =
            McfProblem::new(vec![0], vec![1], vec![0], vec![0], vec![1], vec![-1, 1]).unwrap();
        let err = initialize_feasible_flow(&problem).unwrap_err();
        assert!(matches!(err, McfError::Infeasible));
    }

    #[test]
    fn initializer_produces_strictly_interior_flows_on_random_graphs() {
        let mut rng = Lcg::new(7);
        for _ in 0..5 {
            let n = 4 + rng.next_range(3) as usize;
            let m = 8 + rng.next_range(5) as usize;
            let mut tails = Vec::with_capacity(m);
            let mut heads = Vec::with_capacity(m);
            let mut lower = Vec::with_capacity(m);
            let mut upper = Vec::with_capacity(m);
            let mut cost = Vec::with_capacity(m);
            let mut base_flow = Vec::with_capacity(m);

            for _ in 0..m {
                let tail = rng.next_range(n as u32);
                let mut head = rng.next_range(n as u32);
                if head == tail {
                    head = (head + 1) % n as u32;
                }
                tails.push(tail);
                heads.push(head);
                let lo = rng.next_range(3) as i64;
                let slack = 2 + rng.next_range(3) as i64;
                let up = lo + slack;
                lower.push(lo);
                upper.push(up);
                cost.push(rng.next_range(5) as i64);
                base_flow.push(lo + 1);
            }

            let mut demands = vec![0_i64; n];
            for (idx, &f) in base_flow.iter().enumerate() {
                demands[tails[idx] as usize] -= f;
                demands[heads[idx] as usize] += f;
            }

            let problem = McfProblem::new(tails, heads, lower, upper, cost, demands).unwrap();
            let feasible = initialize_feasible_flow(&problem).unwrap();
            assert_feasible(&problem, &feasible.flow);
            for (idx, &f) in feasible.flow.iter().enumerate() {
                let lo = problem.lower[idx] as f64;
                let up = problem.upper[idx] as f64;
                assert!(
                    f > lo + 1e-9 && f < up - 1e-9,
                    "edge {idx} is not strictly interior: {f} not in ({lo}, {up})"
                );
            }
        }
    }

    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u32(&mut self) -> u32 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.state >> 32) as u32
        }

        fn next_range(&mut self, max: u32) -> u32 {
            if max == 0 {
                0
            } else {
                self.next_u32() % max
            }
        }
    }

    fn cycle_problem(cost: Vec<i64>) -> McfProblem {
        McfProblem::new(
            vec![0, 1, 2],
            vec![1, 2, 0],
            vec![0, 0, 0],
            vec![2, 2, 2],
            cost,
            vec![0, 0, 0],
        )
        .unwrap()
    }

    #[test]
    fn ipm_decreases_potential_monotonically() {
        let problem = cycle_problem(vec![-2, -1, -3]);
        let opts = McfOptions {
            tolerance: 1e-6,
            max_iters: 4,
            ..McfOptions::default()
        };
        let result = run_ipm(&problem, &opts).unwrap();
        let potentials = result.stats.potentials;
        assert!(potentials.len() >= 2);
        for window in potentials.windows(2) {
            assert!(
                window[1] <= window[0] + 1e-9,
                "potential increased from {} to {}",
                window[0],
                window[1]
            );
        }
        assert!(result.stats.last_step_size > 0.0);
    }

    #[test]
    fn ipm_stops_on_iteration_limit() {
        let problem = cycle_problem(vec![-2, -1, -3]);
        let opts = McfOptions {
            tolerance: 1e-12,
            max_iters: 1,
            ..McfOptions::default()
        };
        let result = run_ipm(&problem, &opts).unwrap();
        assert_eq!(result.termination, IpmTermination::IterationLimit);
    }

    #[test]
    fn ipm_respects_time_limit() {
        let problem = cycle_problem(vec![-2, -1, -3]);
        let opts = McfOptions {
            time_limit_ms: Some(0),
            ..McfOptions::default()
        };
        let result = run_ipm(&problem, &opts).unwrap();
        assert_eq!(result.termination, IpmTermination::TimeLimit);
    }

    #[test]
    fn ipm_converges_on_simple_network() {
        let problem = cycle_problem(vec![2, 1, 3]);
        let opts = McfOptions {
            tolerance: 1e-6,
            max_iters: 20,
            ..McfOptions::default()
        };
        let result = run_ipm(&problem, &opts).unwrap();
        assert_eq!(result.termination, IpmTermination::Converged);
    }
}
