use almo_mcf_core::ipm::run_ipm;
use almo_mcf_core::{McfOptions, McfProblem, OracleMode, Strategy};

#[test]
fn test_rebuilding_game() {
    let problem = McfProblem::new(
        vec![0, 1, 0],
        vec![1, 2, 2],
        vec![0, 0, 0],
        vec![5, 5, 5],
        vec![1, 1, 3],
        vec![-5, 0, 5],
    )
    .unwrap();
    let opts = McfOptions {
        strategy: Strategy::FullDynamic,
        oracle_mode: OracleMode::Dynamic,
        max_iters: 20,
        ..McfOptions::default()
    };
    let result = run_ipm(&problem, &opts).unwrap();
    assert!(result.stats.iterations <= opts.max_iters);
}
