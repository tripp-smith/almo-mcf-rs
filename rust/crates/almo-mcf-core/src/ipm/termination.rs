use crate::numerics::barrier::{clamped_barrier_term, BarrierClampConfig};
use crate::McfOptions;

use super::Residuals;

pub type FlowVec = [f64];
pub type CostVec = [f64];
pub type LengthVec = Residuals;

pub const DEFAULT_GAP_EXPONENT: f64 = 10.0;
pub const DEFAULT_GAP_THRESHOLD_FACTOR: f64 = 1e-10;

pub fn compute_barrier_sum(
    flow: &FlowVec,
    residuals: &Residuals,
    alpha: f64,
    clamp_config: &BarrierClampConfig,
) -> f64 {
    if flow.len() != residuals.upper.len() || flow.len() != residuals.lower.len() {
        return f64::INFINITY;
    }
    let mut sum = 0.0;
    for (&upper_res, &lower_res) in residuals.upper.iter().zip(residuals.lower.iter()) {
        let upper_term = clamped_barrier_term(upper_res, alpha, clamp_config, None);
        let lower_term = clamped_barrier_term(lower_res, alpha, clamp_config, None);
        let combined = upper_term + lower_term;
        if !combined.is_finite() {
            return f64::INFINITY;
        }
        sum += combined;
    }
    sum
}

pub fn estimate_duality_gap(
    current_flow: &FlowVec,
    costs: &CostVec,
    lengths: &LengthVec,
    barrier_alpha: f64,
    m: usize,
    u: f64,
) -> f64 {
    estimate_duality_gap_with_config(
        current_flow,
        costs,
        lengths,
        barrier_alpha,
        m,
        u,
        &BarrierClampConfig::default(),
    )
}

pub fn estimate_duality_gap_with_config(
    current_flow: &FlowVec,
    costs: &CostVec,
    residuals: &Residuals,
    barrier_alpha: f64,
    m: usize,
    u: f64,
    clamp_config: &BarrierClampConfig,
) -> f64 {
    if current_flow.len() != costs.len() {
        return f64::INFINITY;
    }
    let base_cost = current_flow
        .iter()
        .zip(costs.iter())
        .map(|(f, c)| f * c)
        .sum::<f64>();
    if !base_cost.is_finite() {
        return f64::INFINITY;
    }
    let gap_scale = base_cost.abs().max(1.0).max(u.max(1.0));
    let log_gap_proxy = gap_scale.ln();
    let barrier_sum = compute_barrier_sum(current_flow, residuals, barrier_alpha, clamp_config);
    if !barrier_sum.is_finite() {
        return f64::INFINITY;
    }
    let phi = (m as f64) * log_gap_proxy + barrier_alpha.abs() * barrier_sum;
    if !phi.is_finite() {
        return f64::INFINITY;
    }
    let denom = (20.0 * m.max(1) as f64).max(1.0);
    let estimate = gap_scale * (-phi / denom).exp();
    if estimate.is_finite() {
        estimate.max(0.0)
    } else {
        f64::INFINITY
    }
}

pub fn compute_gap_threshold(m: usize, max_u: f64, config: &McfOptions) -> f64 {
    if let Some(threshold) = config.gap_threshold {
        return threshold;
    }
    let base = (m as f64 * max_u.max(1.0)).max(1.0);
    base.powf(-config.gap_exponent)
}

pub fn should_terminate(gap: f64, m: usize, max_u: f64, config: &McfOptions) -> bool {
    let threshold = compute_gap_threshold(m, max_u, config);
    eprintln!(
        "gap termination check: gap={:.3e}, threshold={:.3e}",
        gap, threshold
    );
    gap <= threshold
}
