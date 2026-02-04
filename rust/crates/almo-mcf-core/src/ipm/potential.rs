use crate::numerics::barrier::{
    clamped_alpha, guarded_inverse_with_stats, guarded_log_with_stats, guarded_pow_with_stats,
    BarrierClampConfig, BarrierClampStats,
};
use crate::numerics::GRADIENT_EPSILON;

#[derive(Debug, Clone)]
pub struct Residuals {
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
}

impl Residuals {
    pub fn new(flow: &[f64], lower: &[f64], upper: &[f64], min_delta: f64) -> Self {
        Self::new_with_stats(flow, lower, upper, min_delta, None)
    }

    pub fn new_with_stats(
        flow: &[f64],
        lower: &[f64],
        upper: &[f64],
        min_delta: f64,
        mut stats: Option<&mut BarrierClampStats>,
    ) -> Self {
        let mut upper_res = vec![0.0; flow.len()];
        let mut lower_res = vec![0.0; flow.len()];
        for ((u_res, l_res), ((f, l), u)) in upper_res
            .iter_mut()
            .zip(lower_res.iter_mut())
            .zip(flow.iter().zip(lower.iter()).zip(upper.iter()))
        {
            let upper_delta = *u - *f;
            let lower_delta = *f - *l;
            let clamped_upper = upper_delta.max(min_delta);
            let clamped_lower = lower_delta.max(min_delta);
            if clamped_upper != upper_delta || clamped_lower != lower_delta {
                if let Some(stats) = stats.as_mut() {
                    stats.clamped_residuals += 1;
                }
            }
            if let Some(stats) = stats.as_mut() {
                stats.record_residual(clamped_upper);
                stats.record_residual(clamped_lower);
            }
            *u_res = clamped_upper;
            *l_res = clamped_lower;
        }
        Self {
            upper: upper_res,
            lower: lower_res,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn compute_potential(
    cost: &[f64],
    flow: &[f64],
    residuals: &Residuals,
    alpha: f64,
    beta: f64,
    cost_lower_bound: f64,
    min_gap: f64,
    clamp_config: &BarrierClampConfig,
    mut stats: Option<&mut BarrierClampStats>,
) -> f64 {
    let base_cost = cost
        .iter()
        .zip(flow.iter())
        .map(|(c, f)| c * f)
        .sum::<f64>();
    let barrier = residuals
        .upper
        .iter()
        .zip(residuals.lower.iter())
        .map(|(upper, lower)| {
            let upper_pow =
                guarded_pow_with_stats(*upper, beta, clamp_config.max_log, stats.as_deref_mut());
            let lower_pow =
                guarded_pow_with_stats(*lower, beta, clamp_config.max_log, stats.as_deref_mut());
            let combined = (upper_pow + lower_pow).clamp(0.0, clamp_config.barrier_clamp_max);
            if combined != upper_pow + lower_pow {
                if let Some(stats) = stats.as_mut() {
                    stats.clamped_barrier += 1;
                }
            }
            if let Some(stats) = stats.as_mut() {
                stats.record_barrier_value(combined);
            }
            combined
        })
        .sum::<f64>();
    let gap = (base_cost - cost_lower_bound).max(min_gap);
    let log_term = guarded_log_with_stats(gap, min_gap, stats);
    base_cost + (1.0 / alpha) * log_term + barrier
}

#[allow(clippy::too_many_arguments)]
pub fn compute_gradient(
    cost: &[f64],
    flow: &[f64],
    residuals: &Residuals,
    alpha: f64,
    beta: f64,
    cost_lower_bound: f64,
    min_gap: f64,
    clamp_config: &BarrierClampConfig,
    mut stats: Option<&mut BarrierClampStats>,
) -> Vec<f64> {
    let base_cost = cost
        .iter()
        .zip(flow.iter())
        .map(|(c, f)| c * f)
        .sum::<f64>();
    let gap = (base_cost - cost_lower_bound).max(min_gap);
    let log_coeff = 1.0 / alpha;
    let gap_inv = guarded_inverse_with_stats(gap, clamp_config.min_x, stats.as_deref_mut());
    cost.iter()
        .zip(residuals.upper.iter().zip(residuals.lower.iter()))
        .map(|(c, (upper, lower))| {
            let upper_pow = guarded_pow_with_stats(
                *upper,
                beta - 1.0,
                clamp_config.max_log,
                stats.as_deref_mut(),
            );
            let lower_pow = guarded_pow_with_stats(
                *lower,
                beta - 1.0,
                clamp_config.max_log,
                stats.as_deref_mut(),
            );
            let barrier_term = beta * (upper_pow - lower_pow);
            let clamped_barrier_term = barrier_term.clamp(
                -clamp_config.gradient_clamp_max,
                clamp_config.gradient_clamp_max,
            );
            if clamped_barrier_term != barrier_term {
                if let Some(stats) = stats.as_mut() {
                    stats.clamped_gradient += 1;
                }
            }
            c + log_coeff * (c * gap_inv) + clamped_barrier_term
        })
        .collect()
}

pub fn compute_lengths(
    gradient: &[f64],
    clamp_config: &BarrierClampConfig,
    mut stats: Option<&mut BarrierClampStats>,
) -> Vec<f64> {
    gradient
        .iter()
        .map(|g| {
            let denom = g.abs().max(GRADIENT_EPSILON);
            guarded_inverse_with_stats(denom, clamp_config.min_x, stats.as_deref_mut())
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct Potential {
    pub alpha: f64,
    pub beta: f64,
    pub min_delta: f64,
    pub threads: usize,
    pub m_u: f64,
    pub edge_count: f64,
    pub cost_lower_bound: f64,
    pub clamp_config: BarrierClampConfig,
}

impl Potential {
    pub fn new(
        upper: &[f64],
        alpha_override: Option<f64>,
        threads: usize,
        clamp_config: BarrierClampConfig,
    ) -> Self {
        Self::new_with_lower_bound(upper, alpha_override, None, threads, 0.0, clamp_config)
    }

    pub fn new_with_lower_bound(
        upper: &[f64],
        alpha_override: Option<f64>,
        beta_override: Option<f64>,
        threads: usize,
        cost_lower_bound: f64,
        clamp_config: BarrierClampConfig,
    ) -> Self {
        let edge_count = upper.len().max(1) as f64;
        let max_upper = upper
            .iter()
            .cloned()
            .fold(1.0_f64, |acc, value| acc.max(value.abs()).max(1.0));
        let m_u = (edge_count * max_upper).max(1.0);
        let denom = (m_u.log2()).max(1.0);
        let default_alpha = 1.0 / (1000.0 * denom);
        let alpha = clamped_alpha(
            alpha_override
                .filter(|value| *value > 0.0)
                .unwrap_or(default_alpha),
            &clamp_config,
        );
        let beta = beta_override.filter(|value| *value > 0.0).unwrap_or(1.1);
        Self {
            alpha,
            beta,
            min_delta: clamp_config.residual_min,
            threads,
            m_u,
            edge_count,
            cost_lower_bound,
            clamp_config,
        }
    }

    pub fn value(&self, cost: &[f64], flow: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
        let residuals = Residuals::new(flow, lower, upper, self.min_delta);
        self.value_with_residuals(cost, flow, &residuals)
    }

    pub fn value_with_residuals(&self, cost: &[f64], flow: &[f64], residuals: &Residuals) -> f64 {
        self.value_with_residuals_and_stats(cost, flow, residuals, None)
    }

    pub fn value_with_residuals_and_stats(
        &self,
        cost: &[f64],
        flow: &[f64],
        residuals: &Residuals,
        stats: Option<&mut BarrierClampStats>,
    ) -> f64 {
        compute_potential(
            cost,
            flow,
            residuals,
            self.alpha,
            self.beta,
            self.cost_lower_bound,
            self.rounding_threshold(),
            &self.clamp_config,
            stats,
        )
    }

    pub fn lengths_from_gradient(&self, gradient: &[f64]) -> Vec<f64> {
        self.lengths_from_gradient_with_stats(gradient, None)
    }

    pub fn lengths_from_gradient_with_stats(
        &self,
        gradient: &[f64],
        stats: Option<&mut BarrierClampStats>,
    ) -> Vec<f64> {
        compute_lengths(gradient, &self.clamp_config, stats)
    }

    pub fn gradient(&self, cost: &[f64], flow: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
        let residuals = Residuals::new(flow, lower, upper, self.min_delta);
        self.gradient_with_residuals(cost, flow, &residuals)
    }

    pub fn gradient_with_residuals(
        &self,
        cost: &[f64],
        flow: &[f64],
        residuals: &Residuals,
    ) -> Vec<f64> {
        self.gradient_with_residuals_and_stats(cost, flow, residuals, None)
    }

    pub fn gradient_with_residuals_and_stats(
        &self,
        cost: &[f64],
        flow: &[f64],
        residuals: &Residuals,
        stats: Option<&mut BarrierClampStats>,
    ) -> Vec<f64> {
        compute_gradient(
            cost,
            flow,
            residuals,
            self.alpha,
            self.beta,
            self.cost_lower_bound,
            self.rounding_threshold(),
            &self.clamp_config,
            stats,
        )
    }

    pub fn duality_gap(&self, cost: &[f64], flow: &[f64], residuals: &Residuals) -> f64 {
        let base_cost = cost
            .iter()
            .zip(flow.iter())
            .map(|(c, f)| c * f)
            .sum::<f64>();
        let barrier = residuals
            .upper
            .iter()
            .zip(residuals.lower.iter())
            .map(|(upper, lower)| {
                let upper_pow =
                    guarded_pow_with_stats(*upper, self.beta, self.clamp_config.max_log, None);
                let lower_pow =
                    guarded_pow_with_stats(*lower, self.beta, self.clamp_config.max_log, None);
                (upper_pow + lower_pow).clamp(0.0, self.clamp_config.barrier_clamp_max)
            })
            .sum::<f64>();
        let cost_gap = (base_cost - self.cost_lower_bound).max(0.0);
        cost_gap + barrier / self.beta
    }

    pub fn termination_target(&self) -> f64 {
        -200.0 * self.edge_count * self.m_u.ln().max(1.0)
    }

    pub fn reduction_floor(&self, current_potential: f64) -> f64 {
        let log_term = self.m_u.ln().max(2.0);
        let denom = log_term * log_term * log_term * self.edge_count.max(1.0);
        let factor = 1.0 / denom.max(1.0);
        let floor = current_potential.abs().max(1.0) * factor;
        floor.min(1e-6)
    }

    pub fn rounding_threshold(&self) -> f64 {
        self.m_u.powf(-10.0)
    }

    pub fn kappa_floor(&self) -> f64 {
        let log_term = self.edge_count.ln().max(2.0);
        log_term.powf(-1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_alpha_and_thresholds_from_m_u() {
        let upper = vec![10.0, 5.0];
        let potential = Potential::new(&upper, None, 1, BarrierClampConfig::default());
        assert!(potential.alpha > 0.0);
        assert!(potential.rounding_threshold() < 1.0);
        assert!(potential.termination_target().is_sign_negative());
    }

    #[test]
    fn reduction_floor_scales_with_potential() {
        let upper = vec![8.0, 4.0, 6.0];
        let potential = Potential::new(&upper, None, 1, BarrierClampConfig::default());
        let floor_small = potential.reduction_floor(1.0);
        let floor_large = potential.reduction_floor(100.0);
        assert!(floor_small > 0.0);
        assert!(floor_large >= floor_small);
        assert!(floor_small <= 1e-6);
    }

    #[test]
    fn log_term_updates_gradient_with_cost_gap() {
        let upper = vec![10.0, 10.0];
        let lower = vec![0.0, 0.0];
        let flow = vec![2.0, 1.0];
        let cost = vec![3.0, 5.0];
        let base = Potential::new(&upper, None, 1, BarrierClampConfig::default());
        let with_gap = Potential::new_with_lower_bound(
            &upper,
            None,
            None,
            1,
            1.0,
            BarrierClampConfig::default(),
        );

        let base_grad = base.gradient(&cost, &flow, &lower, &upper);
        let base_cost = cost[0] * flow[0] + cost[1] * flow[1];
        let base_gap = (base_cost - base.cost_lower_bound).max(base.rounding_threshold());
        let gap = (base_cost - with_gap.cost_lower_bound).max(with_gap.rounding_threshold());
        let gap_grad = with_gap.gradient(&cost, &flow, &lower, &upper);

        for ((&base_value, &gap_value), &c) in
            base_grad.iter().zip(gap_grad.iter()).zip(cost.iter())
        {
            let expected = base_value + (c / gap - c / base_gap) / with_gap.alpha;
            assert!((gap_value - expected).abs() <= 1e-9);
        }
    }

    #[test]
    fn kappa_floor_is_small_positive() {
        let upper = vec![2.0, 3.0, 5.0];
        let potential = Potential::new(&upper, None, 1, BarrierClampConfig::default());
        let kappa_floor = potential.kappa_floor();
        assert!(kappa_floor > 0.0);
        assert!(kappa_floor < 1.0);
    }

    #[test]
    fn gradient_matches_hand_calculation() {
        let upper = vec![4.0];
        let lower = vec![0.0];
        let flow = vec![1.0];
        let cost = vec![2.0];
        let potential = Potential::new_with_lower_bound(
            &upper,
            Some(0.5),
            Some(0.5),
            1,
            0.0,
            BarrierClampConfig::default(),
        );

        let base_cost: f64 = cost[0] * flow[0];
        let gap = base_cost.max(potential.rounding_threshold());
        let upper_slack = (upper[0] - flow[0]).max(potential.min_delta);
        let lower_slack = (flow[0] - lower[0]).max(potential.min_delta);
        let expected_barrier = 0.5 * (upper_slack.powf(-0.5) - lower_slack.powf(-0.5));
        let expected = cost[0] + (cost[0] / gap) / potential.alpha + expected_barrier;

        let gradient = potential.gradient(&cost, &flow, &lower, &upper);
        assert!((gradient[0] - expected).abs() <= 1e-9);
    }
}
