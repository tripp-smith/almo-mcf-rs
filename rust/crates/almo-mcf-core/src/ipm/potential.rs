use crate::numerics::barrier::{barrier_gradient, barrier_lengths};
use crate::numerics::duality_gap_proxy;

#[derive(Debug, Clone)]
pub struct Potential {
    pub alpha: f64,
    pub min_delta: f64,
    pub threads: usize,
    pub m_u: f64,
    pub edge_count: f64,
}

impl Potential {
    pub fn new(upper: &[f64], alpha_override: Option<f64>, threads: usize) -> Self {
        let edge_count = upper.len().max(1) as f64;
        let max_upper = upper
            .iter()
            .cloned()
            .fold(1.0_f64, |acc, value| acc.max(value.abs()).max(1.0));
        let m_u = (edge_count * max_upper).max(1.0);
        let denom = (m_u.ln()).max(1.0);
        let default_alpha = 1.0 / (1000.0 * denom);
        let alpha = alpha_override
            .filter(|value| *value > 0.0)
            .unwrap_or(default_alpha);
        Self {
            alpha,
            min_delta: 1e-9,
            threads,
            m_u,
            edge_count,
        }
    }

    pub fn value(&self, cost: &[f64], flow: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
        let base_cost = cost
            .iter()
            .zip(flow.iter())
            .map(|(c, f)| c * f)
            .sum::<f64>();
        let barrier = barrier_lengths(flow, lower, upper, self.alpha, self.min_delta, self.threads)
            .iter()
            .sum::<f64>();
        (1.0 + self.alpha) * base_cost + barrier
    }

    pub fn lengths(&self, flow: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
        barrier_lengths(flow, lower, upper, self.alpha, self.min_delta, self.threads)
    }

    pub fn gradient(&self, cost: &[f64], flow: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
        let barrier =
            barrier_gradient(flow, lower, upper, self.alpha, self.min_delta, self.threads);
        cost.iter()
            .zip(barrier.iter())
            .map(|(c, b)| (1.0 + self.alpha) * c + b)
            .collect()
    }

    pub fn duality_gap(&self, gradient: &[f64], flow: &[f64]) -> f64 {
        duality_gap_proxy(gradient, flow)
    }

    pub fn termination_target(&self) -> f64 {
        -200.0 * self.edge_count * self.m_u.ln().max(1.0)
    }

    pub fn reduction_floor(&self, current_potential: f64) -> f64 {
        let log_term = self.m_u.ln().max(2.0);
        let factor = 1.0 / (log_term * log_term);
        current_potential.abs().max(1.0) * factor
    }

    pub fn rounding_threshold(&self) -> f64 {
        self.m_u.powf(-10.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_alpha_and_thresholds_from_m_u() {
        let upper = vec![10.0, 5.0];
        let potential = Potential::new(&upper, None, 1);
        assert!(potential.alpha > 0.0);
        assert!(potential.rounding_threshold() < 1.0);
        assert!(potential.termination_target().is_sign_negative());
    }

    #[test]
    fn reduction_floor_scales_with_potential() {
        let upper = vec![8.0, 4.0, 6.0];
        let potential = Potential::new(&upper, None, 1);
        let floor_small = potential.reduction_floor(1.0);
        let floor_large = potential.reduction_floor(100.0);
        assert!(floor_small > 0.0);
        assert!(floor_large > floor_small);
    }
}
