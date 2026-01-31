use crate::numerics::barrier::barrier_lengths;
use crate::numerics::{duality_gap_proxy, gradient_proxy};

#[derive(Debug, Clone)]
pub struct Potential {
    pub alpha: f64,
    pub min_delta: f64,
    pub threads: usize,
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
        base_cost + self.alpha * base_cost + barrier
    }

    pub fn lengths(&self, flow: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
        barrier_lengths(flow, lower, upper, self.alpha, self.min_delta, self.threads)
    }

    pub fn gradient(&self, cost: &[f64], flow: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
        gradient_proxy(cost, flow, lower, upper, self.alpha, self.min_delta)
    }

    pub fn duality_gap(&self, gradient: &[f64], flow: &[f64]) -> f64 {
        duality_gap_proxy(gradient, flow)
    }
}
