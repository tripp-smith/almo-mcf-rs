use almo_mcf_core::solver::set_dynamic_max_iters;

#[test]
fn test_iteration_optimization() {
    let m = 1500;
    let iters = set_dynamic_max_iters(m, 0.2);
    assert!(iters > 0);
    let bound = ((m as f64).sqrt() * (m as f64).ln().max(1.0) * 10.0) as usize;
    assert!(iters <= bound.max(1_000_000));
}
