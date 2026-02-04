use crate::graph::{Graph, NodeId};
use crate::McfError;

const RESIDUAL_EPS: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct ResidualArc {
    pub to: usize,
    pub capacity: f64,
    pub cost: i64,
    pub rev_idx: usize,
    pub edge_id: usize,
    pub direction: i8,
}

#[derive(Debug, Clone)]
pub struct ResidualGraph {
    pub arcs: Vec<Vec<ResidualArc>>,
    pub demand: Vec<f64>,
    pub flow: Vec<f64>,
}

#[derive(Debug, Default, Clone)]
pub struct CancellationStats {
    pub cycles_canceled: usize,
    pub total_flow_augmented: f64,
    pub total_cost_reduction: f64,
    pub final_gap_proxy: f64,
}

impl ResidualGraph {
    pub fn node_count(&self) -> usize {
        self.arcs.len()
    }

    pub fn edge_count(&self) -> usize {
        self.flow.len()
    }

    pub fn total_demand_imbalance(&self) -> f64 {
        self.demand.iter().map(|v| v.abs()).sum::<f64>()
    }

    pub fn is_epsilon_feasible(&self, epsilon: f64) -> bool {
        self.total_demand_imbalance() <= epsilon
    }

    pub fn flow_is_near_integer(&self, epsilon: f64) -> bool {
        self.flow
            .iter()
            .all(|value| (value - value.round()).abs() <= epsilon)
    }
}

pub fn has_negative_cycle(residual: &ResidualGraph) -> bool {
    find_negative_cycle(residual).is_some()
}

pub fn negative_cycle_cost(residual: &ResidualGraph) -> Option<f64> {
    find_negative_cycle(residual).map(|cycle| {
        cycle
            .iter()
            .map(|(u, idx)| residual.arcs[*u][*idx].cost as f64)
            .sum()
    })
}

pub fn build_residual_graph(
    approx_flow: &[f64],
    original_graph: &Graph,
) -> Result<ResidualGraph, McfError> {
    if approx_flow.len() != original_graph.edge_count() {
        return Err(McfError::InvalidInput(
            "approximate flow length mismatch".to_string(),
        ));
    }
    let node_count = original_graph.node_count();
    let mut arcs = vec![Vec::new(); node_count];
    let mut demand = vec![0.0_f64; node_count];
    let mut flow = vec![0.0_f64; original_graph.edge_count()];

    for (node_idx, value) in demand.iter_mut().enumerate().take(node_count) {
        let node = NodeId(node_idx);
        *value = original_graph
            .demand(node)
            .ok_or_else(|| McfError::InvalidInput("node demand out of range".to_string()))?;
    }

    for (edge_id, edge) in original_graph.edges() {
        let idx = edge_id.0;
        let f = approx_flow[idx];
        if !f.is_finite() {
            return Err(McfError::InvalidInput(
                "approximate flow contains non-finite values".to_string(),
            ));
        }
        let forward_cap = edge.upper - f;
        let backward_cap = f - edge.lower;
        if forward_cap < -RESIDUAL_EPS || backward_cap < -RESIDUAL_EPS {
            return Err(McfError::InvalidInput(
                "approximate flow violates bounds".to_string(),
            ));
        }

        let tail = edge.tail.0;
        let head = edge.head.0;
        demand[tail] -= f;
        demand[head] += f;
        flow[idx] = f;

        let cap_fwd = forward_cap.max(0.0);
        let cap_bwd = backward_cap.max(0.0);
        add_residual_pair(&mut arcs, tail, head, cap_fwd, cap_bwd, edge.cost, idx);
    }

    Ok(ResidualGraph { arcs, demand, flow })
}

fn add_residual_pair(
    arcs: &mut [Vec<ResidualArc>],
    tail: usize,
    head: usize,
    cap_fwd: f64,
    cap_bwd: f64,
    cost: f64,
    edge_id: usize,
) {
    let fwd_idx = arcs[tail].len();
    let bwd_idx = arcs[head].len();
    arcs[tail].push(ResidualArc {
        to: head,
        capacity: cap_fwd,
        cost: cost.round() as i64,
        rev_idx: bwd_idx,
        edge_id,
        direction: 1,
    });
    arcs[head].push(ResidualArc {
        to: tail,
        capacity: cap_bwd,
        cost: (-cost).round() as i64,
        rev_idx: fwd_idx,
        edge_id,
        direction: -1,
    });
}

pub fn cancel_negative_cycles(
    residual: &mut ResidualGraph,
    max_iterations: Option<usize>,
) -> Result<CancellationStats, McfError> {
    let mut stats = CancellationStats::default();
    let limit = max_iterations.unwrap_or(10_000);
    for _ in 0..limit {
        let cycle = find_negative_cycle(residual);
        let Some(cycle) = cycle else {
            stats.final_gap_proxy = 0.0;
            return Ok(stats);
        };
        let mut min_cap = f64::INFINITY;
        let mut cycle_cost = 0_i64;
        for (u, arc_idx) in &cycle {
            let arc = &residual.arcs[*u][*arc_idx];
            min_cap = min_cap.min(arc.capacity);
            cycle_cost = cycle_cost.saturating_add(arc.cost);
        }
        if !min_cap.is_finite() || min_cap <= RESIDUAL_EPS {
            break;
        }

        for (u, arc_idx) in &cycle {
            let (to, rev_idx, edge_id, direction) = {
                let arc = &mut residual.arcs[*u][*arc_idx];
                arc.capacity -= min_cap;
                (arc.to, arc.rev_idx, arc.edge_id, arc.direction)
            };
            let rev_arc = &mut residual.arcs[to][rev_idx];
            rev_arc.capacity += min_cap;
            residual.flow[edge_id] += direction as f64 * min_cap;
        }
        stats.cycles_canceled += 1;
        stats.total_flow_augmented += min_cap;
        if cycle_cost < 0 {
            stats.total_cost_reduction += -min_cap * cycle_cost as f64;
        }
    }

    if let Some(cycle) = find_negative_cycle(residual) {
        stats.final_gap_proxy = cycle
            .iter()
            .map(|(u, idx)| residual.arcs[*u][*idx].cost as f64)
            .sum();
    }
    Ok(stats)
}

fn find_negative_cycle(residual: &ResidualGraph) -> Option<Vec<(usize, usize)>> {
    let n = residual.node_count();
    let mut dist = vec![0.0_f64; n];
    let mut pred: Vec<Option<(usize, usize)>> = vec![None; n];
    let mut updated = None;

    for _ in 0..n {
        updated = None;
        for u in 0..n {
            let du = dist[u];
            for (idx, arc) in residual.arcs[u].iter().enumerate() {
                if arc.capacity <= RESIDUAL_EPS {
                    continue;
                }
                let v = arc.to;
                let nd = du + arc.cost as f64;
                if nd + RESIDUAL_EPS < dist[v] {
                    dist[v] = nd;
                    pred[v] = Some((u, idx));
                    updated = Some(v);
                }
            }
        }
        let updated_node = updated?;
        updated = Some(updated_node);
    }

    let mut v = updated?;
    for _ in 0..n {
        if let Some((u, _)) = pred[v] {
            v = u;
        }
    }
    let start = v;
    let mut cycle = Vec::new();
    let mut current = start;
    loop {
        let (u, idx) = pred[current]?;
        cycle.push((u, idx));
        current = u;
        if current == start {
            break;
        }
    }
    Some(cycle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_residual_demands() {
        let mut graph = Graph::with_demands(vec![-1.0, 1.0]);
        graph.add_edge(NodeId(0), NodeId(1), 0.0, 2.0, 1.0).unwrap();
        let residual = build_residual_graph(&[1.0], &graph).unwrap();
        assert_eq!(residual.demand.len(), 2);
        assert!((residual.demand[0] + 2.0).abs() < 1e-9);
        assert!((residual.demand[1] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn cancels_negative_cycle() {
        let mut graph = Graph::with_demands(vec![0.0, 0.0]);
        graph
            .add_edge(NodeId(0), NodeId(1), 0.0, 2.0, -1.0)
            .unwrap();
        graph.add_edge(NodeId(1), NodeId(0), 0.0, 2.0, 0.0).unwrap();
        let mut residual = build_residual_graph(&[0.0, 0.0], &graph).unwrap();
        let stats = cancel_negative_cycles(&mut residual, Some(10)).unwrap();
        assert!(stats.cycles_canceled >= 1);
        assert!(residual.flow.iter().any(|value| *value > 0.0));
    }
}
