use almo_mcf_core::rounding::{build_residual_instance, round_fractional_flow};
use almo_mcf_core::McfProblem;

#[test]
fn rounding_preserves_bounds_and_costs() {
    let problem = McfProblem::new(
        vec![0, 0, 1],
        vec![1, 2, 2],
        vec![0, 0, 0],
        vec![2, 2, 2],
        vec![3, 1, 2],
        vec![-2, 0, 2],
    )
    .unwrap();
    let fractional = vec![1.5, 0.5, 1.0];
    let rounded = round_fractional_flow(&problem, &fractional).unwrap();
    assert_eq!(rounded.flow.len(), problem.edge_count());
    for (idx, &f) in rounded.flow.iter().enumerate() {
        assert!(f >= problem.lower[idx] && f <= problem.upper[idx]);
    }
    assert!(rounded.cost <= 6);
}

#[test]
fn residual_builder_returns_unit_caps() {
    let problem = McfProblem::new(
        vec![0, 1],
        vec![1, 2],
        vec![0, 0],
        vec![2, 2],
        vec![1, 2],
        vec![-1, 0, 1],
    )
    .unwrap();
    let fractional = vec![0.2, 1.0];
    let residual = build_residual_instance(&problem, &fractional).unwrap();
    assert_eq!(residual.capacity, vec![1]);
    assert_eq!(residual.edge_map, vec![0]);
}
