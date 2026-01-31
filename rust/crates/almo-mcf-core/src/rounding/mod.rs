use crate::graph::min_cost_flow::MinCostFlow;
use crate::{McfError, McfProblem, McfSolution};

const FRACTIONAL_EPS: f64 = 1e-9;

#[derive(Debug, Default, Clone)]
pub struct RoundingPlan {
    pub max_cycles: usize,
}

#[derive(Debug, Clone)]
pub struct ResidualInstance {
    pub base_flow: Vec<i64>,
    pub tails: Vec<u32>,
    pub heads: Vec<u32>,
    pub capacity: Vec<i64>,
    pub cost: Vec<i64>,
    pub demand: Vec<i64>,
    pub edge_map: Vec<usize>,
}

pub fn build_residual_instance(
    problem: &McfProblem,
    fractional_flow: &[f64],
) -> Result<ResidualInstance, McfError> {
    let m = problem.edge_count();
    let n = problem.node_count;
    if fractional_flow.len() != m {
        return Err(McfError::InvalidInput(
            "fractional flow length mismatch".to_string(),
        ));
    }

    let mut base_flow = vec![0_i64; m];
    let mut balance = vec![0_i64; n];
    let mut tails = Vec::new();
    let mut heads = Vec::new();
    let mut capacity = Vec::new();
    let mut cost = Vec::new();
    let mut edge_map = Vec::new();

    for i in 0..m {
        let f = fractional_flow[i];
        if !f.is_finite() {
            return Err(McfError::InvalidInput(
                "fractional flow contains non-finite values".to_string(),
            ));
        }
        let lo = problem.lower[i] as f64;
        let up = problem.upper[i] as f64;
        if f < lo - FRACTIONAL_EPS || f > up + FRACTIONAL_EPS {
            return Err(McfError::InvalidInput(
                "fractional flow violates bounds".to_string(),
            ));
        }
        let rounded = f.round();
        let base = if (f - rounded).abs() <= FRACTIONAL_EPS {
            rounded as i64
        } else {
            f.floor() as i64
        };
        if base < problem.lower[i] || base > problem.upper[i] {
            return Err(McfError::InvalidInput(
                "rounded flow violates bounds".to_string(),
            ));
        }
        base_flow[i] = base;
        let tail = problem.tails[i] as usize;
        let head = problem.heads[i] as usize;
        balance[tail] -= base;
        balance[head] += base;

        let frac = f - base as f64;
        if frac > FRACTIONAL_EPS {
            if base == problem.upper[i] {
                return Err(McfError::InvalidInput(
                    "fractional flow exceeds integer capacity".to_string(),
                ));
            }
            tails.push(problem.tails[i]);
            heads.push(problem.heads[i]);
            capacity.push(1);
            cost.push(problem.cost[i]);
            edge_map.push(i);
        }
    }

    let mut demand = vec![0_i64; n];
    for (idx, &b) in balance.iter().enumerate() {
        demand[idx] = problem.demands[idx]
            .checked_sub(b)
            .ok_or_else(|| McfError::InvalidInput("demand overflow".to_string()))?;
    }
    let demand_sum: i64 = demand.iter().sum();
    if demand_sum != 0 {
        return Err(McfError::InvalidInput(
            "residual demands do not sum to zero".to_string(),
        ));
    }

    Ok(ResidualInstance {
        base_flow,
        tails,
        heads,
        capacity,
        cost,
        demand,
        edge_map,
    })
}

pub fn round_fractional_flow(
    problem: &McfProblem,
    fractional_flow: &[f64],
) -> Result<McfSolution, McfError> {
    let residual = build_residual_instance(problem, fractional_flow)?;
    let m = problem.edge_count();
    let n = problem.node_count;

    if residual.tails.is_empty() {
        let cost = residual
            .base_flow
            .iter()
            .zip(problem.cost.iter())
            .map(|(&f, &c)| f as i128 * c as i128)
            .sum::<i128>();
        return Ok(McfSolution {
            flow: residual.base_flow,
            cost,
            ipm_stats: None,
        });
    }

    let total_nodes = n + 2;
    let source = n;
    let sink = n + 1;
    let mut mcf = MinCostFlow::new(total_nodes);
    let mut edge_refs = Vec::with_capacity(residual.tails.len());

    for (idx, (&tail, &head)) in residual.tails.iter().zip(residual.heads.iter()).enumerate() {
        let cap = residual.capacity[idx];
        let (edge_idx, _rev) = mcf.add_edge(tail as usize, head as usize, cap, residual.cost[idx]);
        edge_refs.push((tail as usize, edge_idx, cap, residual.edge_map[idx]));
    }

    let mut total_demand = 0_i64;
    for (node, &b) in residual.demand.iter().enumerate() {
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

    let mut delta = vec![0_i64; m];
    for (tail, edge_idx, cap, edge_id) in edge_refs {
        let edge = &mcf.graph[tail][edge_idx];
        let used = cap - edge.cap;
        delta[edge_id] = used;
    }

    let mut flow = residual.base_flow;
    for i in 0..m {
        flow[i] = flow[i]
            .checked_add(delta[i])
            .ok_or_else(|| McfError::InvalidInput("flow overflow".to_string()))?;
        if flow[i] < problem.lower[i] || flow[i] > problem.upper[i] {
            return Err(McfError::InvalidInput(
                "rounded flow violates bounds".to_string(),
            ));
        }
    }

    let cost = flow
        .iter()
        .zip(problem.cost.iter())
        .map(|(&f, &c)| f as i128 * c as i128)
        .sum::<i128>();

    Ok(McfSolution {
        flow,
        cost,
        ipm_stats: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_fractional_flow_by_min_cost() {
        let problem = McfProblem::new(
            vec![0, 0],
            vec![1, 1],
            vec![0, 0],
            vec![1, 1],
            vec![5, 1],
            vec![-1, 1],
        )
        .unwrap();
        let fractional = vec![0.5, 0.5];
        let rounded = round_fractional_flow(&problem, &fractional).unwrap();
        assert_eq!(rounded.flow, vec![0, 1]);
        assert_eq!(rounded.cost, 1);
    }

    #[test]
    fn builds_residual_instance_with_unit_caps() {
        let problem = McfProblem::new(
            vec![0, 1, 1],
            vec![1, 2, 2],
            vec![0, 0, 0],
            vec![3, 3, 3],
            vec![2, 1, 3],
            vec![0, 0, 0],
        )
        .unwrap();
        let fractional = vec![1.2, 0.0, 2.7];
        let residual = build_residual_instance(&problem, &fractional).unwrap();
        assert_eq!(residual.capacity, vec![1, 1]);
        assert_eq!(residual.edge_map, vec![0, 2]);
        let demand_sum: i64 = residual.demand.iter().sum();
        assert_eq!(demand_sum, 0);
    }
}
