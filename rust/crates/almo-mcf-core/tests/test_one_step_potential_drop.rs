use almo_mcf_core::ipm::{one_step_analysis, Potential};

#[test]
fn test_one_step_potential_drop() {
    let gradients = vec![1.0, -2.0, 0.5];
    let (delta, theta) = one_step_analysis(&gradients, &[(0, 1), (1, -1)], gradients.len());
    assert_eq!(delta.len(), gradients.len());
    assert!(theta > 0.0);

    let upper = vec![2.0, 2.0, 2.0];
    let lower = vec![0.0, 0.0, 0.0];
    let flow = vec![1.0, 1.0, 1.0];
    let cost = vec![1.0, 1.0, 1.0];
    let pot = Potential::new(
        &upper,
        None,
        1,
        almo_mcf_core::McfOptions::default().barrier_clamp_config(),
    );
    let before = pot.value(&cost, &flow, &lower, &upper);
    let candidate: Vec<f64> = flow
        .iter()
        .zip(delta.iter())
        .map(|(f, d)| (f + 0.01 * d).clamp(0.001, 1.999))
        .collect();
    let after = pot.value(&cost, &candidate, &lower, &upper);
    assert!(before.is_finite() && after.is_finite());
}
