use almo_mcf_core::ipm;
use almo_mcf_core::numerics::barrier::{
    clamped_alpha, clamped_barrier_term, clamped_gradient_term, guarded_inverse, guarded_log,
    guarded_pow, BarrierClampConfig,
};
use almo_mcf_core::{McfOptions, McfProblem};
use proptest::prelude::*;

#[test]
fn guarded_math_handles_extremes() {
    let max_log = 700.0;
    assert!(!guarded_pow(1e-300, -10.0, max_log).is_nan());
    assert!(!guarded_pow(1e300, 10.0, max_log).is_nan());
    assert_eq!(guarded_log(0.0, 1e-300), f64::NEG_INFINITY);
    assert!(guarded_log(1e-300, 1e-300).is_finite());
    assert!(guarded_inverse(1e-300, 1e-12).is_finite());
}

#[test]
fn clamped_terms_remain_finite() {
    let config = BarrierClampConfig::default();
    let alpha = clamped_alpha(1e-8, &config);
    let residuals = [1e-30, 1e-12, 1.0, 1e12];
    for &residual in &residuals {
        let barrier = clamped_barrier_term(residual, alpha, &config, None);
        let gradient = clamped_gradient_term(residual, alpha, &config, None);
        assert!(barrier.is_finite());
        assert!(gradient.is_finite());
        assert!(barrier <= config.barrier_clamp_max);
        assert!(gradient.abs() <= config.gradient_clamp_max);
    }
}

#[test]
fn ipm_handles_extreme_costs_and_caps() {
    let problem = McfProblem::new(
        vec![0],
        vec![1],
        vec![0],
        vec![1_000_000_000_000],
        vec![1_000_000_000],
        vec![-1, 1],
    )
    .expect("problem build");
    let opts = McfOptions {
        max_iters: 5,
        tolerance: 1e-9,
        ..McfOptions::default()
    };
    let result = ipm::run_ipm(&problem, &opts).expect("ipm run");
    assert!(result.stats.last_gap.is_finite());
    for value in result.flow {
        assert!(value.is_finite());
    }
}

proptest! {
    #[test]
    fn barrier_terms_stay_finite_in_extreme_ranges(
        residual_exp in -300f64..300f64,
        alpha in 1e-8f64..10.0f64,
    ) {
        let config = BarrierClampConfig::default();
        let residual = 10f64.powf(residual_exp);
        let barrier = clamped_barrier_term(residual, alpha, &config, None);
        let gradient = clamped_gradient_term(residual, alpha, &config, None);
        prop_assert!(barrier.is_finite());
        prop_assert!(gradient.is_finite());
        prop_assert!(barrier <= config.barrier_clamp_max);
        prop_assert!(gradient.abs() <= config.gradient_clamp_max);
    }
}
