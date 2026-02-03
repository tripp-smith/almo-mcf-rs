use almo_mcf_core::ipm::{run_ipm, IpmTermination};
use almo_mcf_core::{McfOptions, McfProblem, Strategy};

fn assert_feasible(problem: &McfProblem, flow: &[f64]) {
    let mut balance = vec![0.0_f64; problem.node_count];
    for (idx, (&f, (&tail, &head))) in flow
        .iter()
        .zip(problem.tails.iter().zip(problem.heads.iter()))
        .enumerate()
    {
        let lo = problem.lower[idx] as f64;
        let up = problem.upper[idx] as f64;
        assert!(f >= lo - 1e-9 && f <= up + 1e-9);
        balance[tail as usize] -= f;
        balance[head as usize] += f;
    }
    for (node, (&b, &d)) in balance.iter().zip(problem.demands.iter()).enumerate() {
        assert!((b - d as f64).abs() <= 1e-9, "node {node} balance mismatch");
    }
}

#[test]
fn ipm_solves_small_cycle() {
    let problem = McfProblem::new(
        vec![0, 1, 2],
        vec![1, 2, 0],
        vec![0, 0, 0],
        vec![3, 3, 3],
        vec![2, -1, 0],
        vec![0, 0, 0],
    )
    .unwrap();
    let opts = McfOptions {
        tolerance: 1e-6,
        max_iters: 50,
        strategy: Strategy::PeriodicRebuild { rebuild_every: 3 },
        ..McfOptions::default()
    };
    let result = run_ipm(&problem, &opts).unwrap();
    assert!(matches!(
        result.termination,
        IpmTermination::Converged | IpmTermination::NoImprovingCycle
    ));
    assert_feasible(&problem, &result.flow);
}

#[test]
fn ipm_full_dynamic_cycle_query_runs() {
    let problem = McfProblem::new(
        vec![0, 1, 2],
        vec![1, 2, 0],
        vec![0, 0, 0],
        vec![3, 3, 3],
        vec![2, -1, 0],
        vec![0, 0, 0],
    )
    .unwrap();
    let opts = McfOptions {
        tolerance: 1e-6,
        max_iters: 25,
        strategy: Strategy::FullDynamic,
        ..McfOptions::default()
    };
    let result = run_ipm(&problem, &opts).unwrap();
    assert!(matches!(
        result.termination,
        IpmTermination::Converged | IpmTermination::NoImprovingCycle
    ));
    assert_feasible(&problem, &result.flow);
}
