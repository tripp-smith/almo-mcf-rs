use almo_mcf_core::solver::run_full_ipm;
use almo_mcf_core::{McfOptions, McfProblem};

#[test]
fn integration_enhanced_ipm() {
    let problem = McfProblem::new(
        vec![0, 0, 1, 2],
        vec![1, 2, 2, 3],
        vec![0, 0, 0, 0],
        vec![5, 5, 5, 5],
        vec![2, 1, 1, 1],
        vec![-4, 0, 0, 4],
    )
    .unwrap();
    let sol = run_full_ipm(&problem, &McfOptions::default()).unwrap();
    assert_eq!(sol.flow.len(), problem.edge_count());
}
