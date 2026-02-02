use crate::numerics::barrier::{barrier_gradient, barrier_lengths, safe_log};
use crate::numerics::duality_gap_proxy;

#[derive(Debug, Clone)]
pub struct Potential {
    pub alpha: f64,
    pub beta: f64,
    pub min_delta: f64,
    pub threads: usize,
    pub m_u: f64,
    pub edge_count: f64,
    pub cost_lower_bound: f64,
}

impl Potential {
    pub fn new(upper: &[f64], alpha_override: Option<f64>, threads: usize) -> Self {
        Self::new_with_lower_bound(upper, alpha_override, None, threads, 0.0)
    }

    pub fn new_with_lower_bound(
        upper: &[f64],
        alpha_override: Option<f64>,
        beta_override: Option<f64>,
        threads: usize,
        cost_lower_bound: f64,
    ) -> Self {
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
        let beta = beta_override.filter(|value| *value > 0.0).unwrap_or(0.5);
        Self {
            alpha,
            beta,
            min_delta: 1e-9,
            threads,
            m_u,
            edge_count,
            cost_lower_bound,
        }
    }

    pub fn value(&self, cost: &[f64], flow: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
        let base_cost = cost
            .iter()
            .zip(flow.iter())
            .map(|(c, f)| c * f)
            .sum::<f64>();
        let barrier = barrier_lengths(flow, lower, upper, self.beta, self.min_delta, self.threads)
            .iter()
            .sum::<f64>();
        let gap = (base_cost - self.cost_lower_bound).max(self.rounding_threshold());
        // Lemma 4.1 (Paper 1): use a log term on the cost gap and power barriers
        // to drive potential reduction while staying strictly interior.
        base_cost + self.alpha * safe_log(gap, self.rounding_threshold()) + barrier
    }

    pub fn lengths(&self, flow: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
        barrier_lengths(flow, lower, upper, self.beta, self.min_delta, self.threads)
    }

    pub fn gradient(&self, cost: &[f64], flow: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
        let barrier = barrier_gradient(flow, lower, upper, self.beta, self.min_delta, self.threads);
        let base_cost = cost
            .iter()
            .zip(flow.iter())
            .map(|(c, f)| c * f)
            .sum::<f64>();
        let gap = (base_cost - self.cost_lower_bound).max(self.rounding_threshold());
        cost.iter()
            .zip(barrier.iter())
            .map(|(c, b)| {
                debug_assert!(gap.is_sign_positive());
                c + self.alpha * (c / gap) + b
            })
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
        log_term.powf(-0.5)
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
        assert!(floor_large >= floor_small);
        assert!(floor_small <= 1e-6);
    }

    #[test]
    fn log_term_updates_gradient_with_cost_gap() {
        let upper = vec![10.0, 10.0];
        let lower = vec![0.0, 0.0];
        let flow = vec![2.0, 1.0];
        let cost = vec![3.0, 5.0];
        let base = Potential::new(&upper, None, 1);
        let with_gap = Potential::new_with_lower_bound(&upper, None, None, 1, 1.0);

        let base_grad = base.gradient(&cost, &flow, &lower, &upper);
        let base_cost = cost[0] * flow[0] + cost[1] * flow[1];
        let base_gap = (base_cost - base.cost_lower_bound).max(base.rounding_threshold());
        let gap = (base_cost - with_gap.cost_lower_bound).max(with_gap.rounding_threshold());
        let gap_grad = with_gap.gradient(&cost, &flow, &lower, &upper);

        for ((&base_value, &gap_value), &c) in
            base_grad.iter().zip(gap_grad.iter()).zip(cost.iter())
        {
            let expected = base_value + with_gap.alpha * (c / gap - c / base_gap);
            assert!((gap_value - expected).abs() <= 1e-9);
        }
    }

    #[test]
    fn kappa_floor_is_small_positive() {
        let upper = vec![2.0, 3.0, 5.0];
        let potential = Potential::new(&upper, None, 1);
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
        let potential = Potential::new_with_lower_bound(&upper, Some(0.5), Some(0.5), 1, 0.0);

        let base_cost: f64 = cost[0] * flow[0];
        let gap = base_cost.max(potential.rounding_threshold());
        let upper_slack = (upper[0] - flow[0]).max(potential.min_delta);
        let lower_slack = (flow[0] - lower[0]).max(potential.min_delta);
        let expected_barrier = 0.5 * upper_slack.powf(-1.5) - 0.5 * lower_slack.powf(-1.5);
        let expected = cost[0] + potential.alpha * (cost[0] / gap) + expected_barrier;

        let gradient = potential.gradient(&cost, &flow, &lower, &upper);
        assert!((gradient[0] - expected).abs() <= 1e-9);
    }
}
