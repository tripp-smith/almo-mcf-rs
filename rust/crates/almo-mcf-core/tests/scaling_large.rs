use almo_mcf_core::scaling::{build_capacity_scaled_problem, build_cost_scaled_problem};
use almo_mcf_core::McfProblem;

fn make_problem(node_count: usize, edge_count: usize, capacity: i64, cost: i64) -> McfProblem {
    let mut tails = Vec::with_capacity(edge_count);
    let mut heads = Vec::with_capacity(edge_count);
    let mut lower = Vec::with_capacity(edge_count);
    let mut upper = Vec::with_capacity(edge_count);
    let mut costs = Vec::with_capacity(edge_count);
    for i in 0..edge_count {
        let tail = i % node_count;
        let head = (i + 1) % node_count;
        tails.push(tail as u32);
        heads.push(head as u32);
        lower.push(0);
        upper.push(capacity + i as i64);
        costs.push(cost + i as i64);
    }
    let demand = vec![0_i64; node_count];
    McfProblem::new(tails, heads, lower, upper, costs, demand).unwrap()
}

#[test]
fn cost_scaling_handles_large_costs() {
    let problem = make_problem(50, 100, 1 << 20, 1_i64 << 40);
    let (_scaled, info) = build_cost_scaled_problem(&problem).unwrap();
    assert!(info.max_scaled_cost <= info.bound);
}

#[test]
fn capacity_scaling_handles_large_capacity() {
    let problem = make_problem(50, 100, 1_i64 << 45, 17);
    let (_scaled, info) = build_capacity_scaled_problem(&problem).unwrap();
    assert!(info.max_scaled_capacity <= info.bound);
}

#[test]
fn scaling_handles_near_i64_limits() {
    let problem = make_problem(10, 20, 1_i64 << 60, 1_i64 << 50);
    let (_scaled_cost, cost_info) = build_cost_scaled_problem(&problem).unwrap();
    let (_scaled_cap, cap_info) = build_capacity_scaled_problem(&problem).unwrap();
    assert!(cost_info.max_scaled_cost <= cost_info.bound);
    assert!(cap_info.max_scaled_capacity <= cap_info.bound);
}
