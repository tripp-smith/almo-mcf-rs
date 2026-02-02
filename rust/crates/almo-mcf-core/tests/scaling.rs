use almo_mcf_core::scaling::{
    build_capacity_scaled_problem, build_cost_scaled_problem, solve_mcf_with_scaling,
};
use almo_mcf_core::{McfOptions, McfProblem};

fn make_large_cost_problem() -> McfProblem {
    let node_count = 9;
    let edge_count = 13;
    let mut tails = Vec::with_capacity(edge_count);
    let mut heads = Vec::with_capacity(edge_count);
    let mut lower = Vec::with_capacity(edge_count);
    let mut upper = Vec::with_capacity(edge_count);
    let mut cost = Vec::with_capacity(edge_count);
    for i in 0..edge_count {
        let tail = i % node_count;
        let head = (i + 1) % node_count;
        tails.push(tail as u32);
        heads.push(head as u32);
        lower.push(0);
        upper.push((1_i64 << 20) + if i % 2 == 0 { 1 } else { 3 });
        cost.push((1_i64 << 40) + (i as i64));
    }
    let demand = vec![0_i64; node_count];
    McfProblem::new(tails, heads, lower, upper, cost, demand).unwrap()
}

#[test]
fn cost_scaling_reduces_cost_bounds() {
    let problem = make_large_cost_problem();
    let (_scaled, info) = build_cost_scaled_problem(&problem).unwrap();
    assert!(info.max_scaled_cost <= info.bound);
}

#[test]
fn capacity_scaling_reduces_capacity_bounds() {
    let node_count = 9;
    let edge_count = 13;
    let mut tails = Vec::with_capacity(edge_count);
    let mut heads = Vec::with_capacity(edge_count);
    let mut lower = Vec::with_capacity(edge_count);
    let mut upper = Vec::with_capacity(edge_count);
    let mut cost = Vec::with_capacity(edge_count);
    for i in 0..edge_count {
        let tail = i % node_count;
        let head = (i + 1) % node_count;
        tails.push(tail as u32);
        heads.push(head as u32);
        lower.push(0);
        upper.push(1_i64 << 20);
        cost.push(1);
    }
    let demand = vec![0_i64; node_count];
    let problem = McfProblem::new(tails, heads, lower, upper, cost, demand).unwrap();

    let (_scaled, info) = build_capacity_scaled_problem(&problem).unwrap();
    assert!(info.max_scaled_capacity <= info.bound);
    assert!(info.divisor >= 1);
}

#[test]
fn solve_with_scaling_keeps_zero_flow_optimum() {
    let problem = make_large_cost_problem();
    let solution = solve_mcf_with_scaling(&problem, &McfOptions::default()).unwrap();
    assert!(solution.flow.iter().all(|&value| value == 0));
    assert_eq!(solution.cost, 0);
}
