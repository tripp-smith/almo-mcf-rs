use almo_mcf_core::numerics::barrier::{safe_exp, safe_log};
use almo_mcf_core::numerics::{
    dot, duality_gap_proxy, gradient_proxy, l1_norm, l2_norm, scaled_add,
};

#[test]
fn vector_ops_compute_expected_norms() {
    let lhs = vec![1.0, -2.0, 3.0];
    let rhs = vec![4.0, 0.5, -1.0];
    assert!((dot(&lhs, &rhs) - 0.0).abs() < 1e-12);
    assert!((l1_norm(&lhs) - 6.0).abs() < 1e-12);
    assert!((l2_norm(&lhs) - (14.0_f64).sqrt()).abs() < 1e-12);
}

#[test]
fn scaled_add_updates_target() {
    let mut target = vec![1.0, 2.0, 3.0];
    let values = vec![0.5, -1.0, 2.0];
    scaled_add(&mut target, 2.0, &values);
    assert_eq!(target, vec![2.0, 0.0, 7.0]);
}

#[test]
fn gradient_proxy_matches_manual_formula() {
    let cost = vec![1.0, -2.0];
    let flow = vec![2.0, 3.0];
    let lower = vec![0.0, 1.0];
    let upper = vec![5.0, 6.0];
    let mu = 0.5;
    let grad = gradient_proxy(&cost, &flow, &lower, &upper, mu, 1e-9);
    let expected0 = 1.0 + mu * (1.0 / (5.0 - 2.0) - 1.0 / (2.0 - 0.0));
    let expected1 = -2.0 + mu * (1.0 / (6.0 - 3.0) - 1.0 / (3.0 - 1.0));
    assert!((grad[0] - expected0).abs() < 1e-12);
    assert!((grad[1] - expected1).abs() < 1e-12);
}

#[test]
fn duality_gap_proxy_is_positive() {
    let gradient = vec![1.0, -2.0, 3.0];
    let flow = vec![0.5, 0.0, 2.0];
    let gap = duality_gap_proxy(&gradient, &flow);
    assert!(gap > 0.0);
}

#[test]
fn safe_log_and_exp_clamp_values() {
    let log_value = safe_log(0.0, 1e-6);
    assert!((log_value - (1e-6_f64).ln()).abs() < 1e-12);
    let exp_value = safe_exp(1000.0, 10.0);
    assert!((exp_value - 10.0_f64.exp()).abs() < 1e-12);
}
