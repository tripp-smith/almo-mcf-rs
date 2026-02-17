use almo_mcf_core::{run_repro_check, McfProblem};

fn fixed_graph() -> McfProblem {
    McfProblem::new(
        vec![0, 0, 1, 1, 2, 2],
        vec![1, 2, 2, 3, 3, 0],
        vec![0, 0, 0, 0, 0, 0],
        vec![8, 8, 8, 8, 8, 8],
        vec![1, 3, 1, 2, 1, 2],
        vec![-6, 0, 0, 6],
    )
    .unwrap()
}

#[test]
fn test_full_pipeline_determinism() {
    let g = fixed_graph();
    for _ in 0..30 {
        assert!(run_repro_check(&g, 42, 42));
    }
}
