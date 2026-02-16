use almo_mcf_core::{run_with_rebuilding_bench, McfOptions, McfProblem};

#[test]
fn integration_rebuilding_mcf() {
    let tails = vec![0, 0, 2, 2, 3];
    let heads = vec![2, 3, 1, 3, 1];
    let lower = vec![0; 5];
    let upper = vec![20; 5];
    let cost = vec![1, 2, 1, 1, 1];
    let demands = vec![-10, 10, 0, 0];
    let problem = McfProblem::new(tails, heads, lower, upper, cost, demands).unwrap();
    let opts = McfOptions {
        use_ipm: Some(true),
        ..McfOptions::default()
    };
    let stats = run_with_rebuilding_bench(&problem, &opts).unwrap();
    assert!(stats.final_gap.is_finite());
}
