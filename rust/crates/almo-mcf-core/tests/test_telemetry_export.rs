use almo_mcf_core::{min_cost_flow_exact, McfOptions, McfProblem};

fn sample_problem() -> McfProblem {
    McfProblem::new(
        vec![0, 0, 1, 2, 1],
        vec![1, 2, 2, 3, 3],
        vec![0, 0, 0, 0, 0],
        vec![10, 10, 10, 10, 10],
        vec![1, 2, 1, 3, 1],
        vec![-5, 0, 0, 5],
    )
    .unwrap()
}

#[test]
fn test_telemetry_export() {
    let problem = sample_problem();
    for _ in 0..20 {
        let sol = min_cost_flow_exact(
            &problem,
            &McfOptions {
                use_ipm: Some(true),
                max_iters: 50,
                ..McfOptions::default()
            },
        )
        .unwrap();
        if let Some(stats) = sol.ipm_stats {
            let rebuild_sum: usize = stats.rebuild_triggers.values().sum();
            let bound = ((problem.node_count.max(2) as f64).ln().ceil() as usize)
                .saturating_mul(stats.total_iters.max(1));
            assert!(rebuild_sum <= bound.max(1));
            let clamp_ratio = if stats.total_iters == 0 {
                0.0
            } else {
                stats.numerical_clamps_applied as f64 / stats.total_iters as f64
            };
            assert!(clamp_ratio <= 0.05 || stats.total_iters < 20);
        }
    }
}
