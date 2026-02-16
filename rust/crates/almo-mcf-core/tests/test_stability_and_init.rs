use almo_mcf_core::ipm::{enforce_stability_bounds, find_initial_point};
use almo_mcf_core::McfProblem;

#[test]
fn test_stability_and_init() {
    let problem = McfProblem::new(
        vec![0, 1, 0],
        vec![1, 2, 2],
        vec![0, 0, 0],
        vec![5, 5, 5],
        vec![1, 2, 3],
        vec![-3, 0, 3],
    )
    .unwrap();
    let (mut x, _phi0) = find_initial_point(&problem);
    let lower: Vec<f64> = problem.lower.iter().map(|&v| v as f64).collect();
    let upper: Vec<f64> = problem.upper.iter().map(|&v| v as f64).collect();
    let changed = enforce_stability_bounds(&mut x, &lower, &upper, 1e-3);
    assert!(changed || !x.is_empty());
    for i in 0..x.len() {
        assert!(x[i] >= lower[i] + 1e-3 - 1e-9);
        assert!(x[i] <= upper[i] - 1e-3 + 1e-9);
    }
}
