use almo_mcf_core::{
    ipm, min_cost_flow_exact, McfOptions, McfProblem, OracleMode, SolverMode, Strategy,
};

fn build_large_zero_cost_problem() -> McfProblem {
    let node_count = 9;
    let mut tails = Vec::new();
    let mut heads = Vec::new();
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    let mut cost = Vec::new();
    for i in 0..13 {
        let tail = i % node_count;
        let head = (i + 1) % node_count;
        tails.push(tail as u32);
        heads.push(head as u32);
        lower.push(0);
        upper.push(5);
        cost.push(0);
    }
    let demand = vec![0_i64; node_count];
    McfProblem::new(tails, heads, lower, upper, cost, demand).unwrap()
}

fn build_large_negative_cost_problem() -> McfProblem {
    let node_count = 9;
    let mut tails = Vec::new();
    let mut heads = Vec::new();
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    let mut cost = Vec::new();
    for i in 0..13 {
        let tail = i % node_count;
        let head = (i + 1) % node_count;
        tails.push(tail as u32);
        heads.push(head as u32);
        lower.push(0);
        upper.push(3);
        cost.push(-1);
    }
    let demand = vec![0_i64; node_count];
    McfProblem::new(tails, heads, lower, upper, cost, demand).unwrap()
}

#[test]
fn terminates_by_gap_with_loose_threshold() {
    let problem = build_large_zero_cost_problem();
    let opts = McfOptions {
        gap_threshold: Some(1e6),
        tolerance: 0.0,
        max_iters: 50,
        strategy: Strategy::FullDynamic,
        ..McfOptions::default()
    };
    let solution = min_cost_flow_exact(&problem, &opts).unwrap();
    let stats = solution.ipm_stats.expect("expected IPM stats");
    assert!(stats.terminated_by_gap);
    assert_eq!(solution.solver_mode, SolverMode::Ipm);
    assert_eq!(solution.cost, 0);

    let classic = min_cost_flow_exact(
        &problem,
        &McfOptions {
            use_ipm: Some(false),
            ..McfOptions::default()
        },
    )
    .unwrap();
    assert_eq!(solution.cost, classic.cost);
}

#[test]
fn terminates_by_max_iters_and_falls_back() {
    let problem = build_large_negative_cost_problem();
    let opts = McfOptions {
        max_iters: 1,
        tolerance: -1.0,
        gap_threshold: Some(1e-30),
        strategy: Strategy::PeriodicRebuild { rebuild_every: 2 },
        oracle_mode: OracleMode::Fallback,
        ..McfOptions::default()
    };
    let solution = min_cost_flow_exact(&problem, &opts).unwrap();
    let stats = solution.ipm_stats.expect("expected IPM stats");
    assert!(stats.terminated_by_max_iters);
    assert_eq!(solution.solver_mode, SolverMode::ClassicFallback);
}

#[test]
fn gap_threshold_handles_large_u() {
    let opts = McfOptions::default();
    let threshold = ipm::compute_gap_threshold(1, 1e18, &opts);
    assert!(threshold.is_finite());
    assert!(threshold > 0.0);
}

#[test]
fn gap_threshold_matches_trivial_case() {
    let opts = McfOptions::default();
    let threshold = ipm::compute_gap_threshold(1, 1.0, &opts);
    assert!((threshold - 1.0).abs() <= 1e-12);
}
