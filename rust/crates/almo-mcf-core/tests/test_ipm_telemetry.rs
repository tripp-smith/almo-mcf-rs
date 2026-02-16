use almo_mcf_core::ipm::verify_almost_linear_iters;
use almo_mcf_core::{min_cost_flow_exact, McfOptions, McfProblem};

#[test]
fn test_ipm_telemetry() {
    let problem = McfProblem::new(
        vec![0, 1, 2, 0],
        vec![1, 2, 0, 2],
        vec![0, 0, 0, 0],
        vec![3, 3, 3, 3],
        vec![1, 1, 1, 1],
        vec![0, 0, 0],
    )
    .unwrap();
    let opts = McfOptions {
        use_ipm: Some(true),
        max_iters: 30,
        ..McfOptions::default()
    };
    let sol = min_cost_flow_exact(&problem, &opts).unwrap();
    if let Some(stats) = sol.ipm_stats {
        assert!(stats.total_iters <= opts.max_iters);
        // weak check on helper path
        let raw = almo_mcf_core::ipm::IpmStats {
            iterations: stats.iterations,
            ..Default::default()
        };
        assert!(verify_almost_linear_iters(problem.edge_count(), &raw));
    }
}
