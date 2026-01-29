use crate::graph::min_cost_flow::MinCostFlow;
use crate::{McfError, McfProblem};

#[derive(Debug, Default, Clone)]
pub struct IpmState {
    pub flow: Vec<f64>,
    pub potential: f64,
}

#[derive(Debug, Default, Clone)]
pub struct IpmStats {
    pub iterations: usize,
    pub last_step_size: f64,
}

#[derive(Debug, Clone)]
pub struct FeasibleFlow {
    pub flow: Vec<i64>,
}

pub fn initialize_feasible_flow(problem: &McfProblem) -> Result<FeasibleFlow, McfError> {
    let n = problem.node_count;
    let m = problem.edge_count();

    let mut demand = problem.demands.clone();
    let mut residual_upper = vec![0_i64; m];
    for (i, residual) in residual_upper.iter_mut().enumerate() {
        let lo = problem.lower[i];
        let up = problem.upper[i];
        if lo > up {
            return Err(McfError::InvalidInput(
                "lower bound exceeds upper bound".to_string(),
            ));
        }
        *residual = up - lo;
        let tail = problem.tails[i] as usize;
        let head = problem.heads[i] as usize;
        demand[tail] = demand[tail].saturating_add(lo);
        demand[head] = demand[head].saturating_sub(lo);
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
        let tail = problem.tails[i] as usize;
        let head = problem.heads[i] as usize;
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
        flow[i] = problem.lower[i] + used;
    }

    Ok(FeasibleFlow { flow })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_feasible(problem: &McfProblem, flow: &[i64]) {
        assert_eq!(flow.len(), problem.edge_count());
        let mut balance = vec![0_i64; problem.node_count];
        for (idx, (&f, (&tail, &head))) in flow
            .iter()
            .zip(problem.tails.iter().zip(problem.heads.iter()))
            .enumerate()
        {
            let lo = problem.lower[idx];
            let up = problem.upper[idx];
            assert!(
                f >= lo && f <= up,
                "flow out of bounds at edge {idx}: {f} not in [{lo}, {up}]"
            );
            balance[tail as usize] -= f;
            balance[head as usize] += f;
        }
        for (node, (&b, &d)) in balance.iter().zip(problem.demands.iter()).enumerate() {
            assert_eq!(
                b, d,
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
}
