pub const DEFAULT_MAX_LOG: f64 = 700.0;
pub const MIN_POSITIVE: f64 = 1e-300;
pub const BARRIER_ALPHA_MIN: f64 = 1e-6;
pub const BARRIER_ALPHA_MAX: f64 = 1.0;
pub const RESIDUAL_MIN: f64 = 1e-12;
pub const BARRIER_CLAMP_MAX: f64 = 1e200;
pub const GRADIENT_CLAMP_MAX: f64 = 1e100;

#[derive(Debug, Clone)]
pub struct BarrierClampConfig {
    pub max_log: f64,
    pub min_x: f64,
    pub residual_min: f64,
    pub barrier_clamp_max: f64,
    pub gradient_clamp_max: f64,
    pub alpha_min: f64,
    pub alpha_max: f64,
}

impl Default for BarrierClampConfig {
    fn default() -> Self {
        Self {
            max_log: DEFAULT_MAX_LOG,
            min_x: MIN_POSITIVE,
            residual_min: RESIDUAL_MIN,
            barrier_clamp_max: BARRIER_CLAMP_MAX,
            gradient_clamp_max: GRADIENT_CLAMP_MAX,
            alpha_min: BARRIER_ALPHA_MIN,
            alpha_max: BARRIER_ALPHA_MAX,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BarrierClampStats {
    pub clamped_logs: usize,
    pub clamped_inverses: usize,
    pub clamped_powers: usize,
    pub clamped_residuals: usize,
    pub clamped_barrier: usize,
    pub clamped_gradient: usize,
    pub clamped_alpha: usize,
    pub max_barrier_value: f64,
    pub min_residual_seen: f64,
}

impl Default for BarrierClampStats {
    fn default() -> Self {
        Self {
            clamped_logs: 0,
            clamped_inverses: 0,
            clamped_powers: 0,
            clamped_residuals: 0,
            clamped_barrier: 0,
            clamped_gradient: 0,
            clamped_alpha: 0,
            max_barrier_value: 0.0,
            min_residual_seen: f64::INFINITY,
        }
    }
}

impl BarrierClampStats {
    pub fn record_residual(&mut self, residual: f64) {
        if residual < self.min_residual_seen {
            self.min_residual_seen = residual;
        }
    }

    pub fn record_barrier_value(&mut self, value: f64) {
        if value.is_finite() && value > self.max_barrier_value {
            self.max_barrier_value = value;
        }
    }

    pub fn total_clamps(&self) -> usize {
        self.clamped_logs
            + self.clamped_inverses
            + self.clamped_powers
            + self.clamped_residuals
            + self.clamped_barrier
            + self.clamped_gradient
            + self.clamped_alpha
    }

    pub fn clamping_occurred(&self) -> bool {
        self.total_clamps() > 0
    }

    pub fn merge(&mut self, other: &Self) {
        self.clamped_logs += other.clamped_logs;
        self.clamped_inverses += other.clamped_inverses;
        self.clamped_powers += other.clamped_powers;
        self.clamped_residuals += other.clamped_residuals;
        self.clamped_barrier += other.clamped_barrier;
        self.clamped_gradient += other.clamped_gradient;
        self.clamped_alpha += other.clamped_alpha;
        if other.max_barrier_value > self.max_barrier_value {
            self.max_barrier_value = other.max_barrier_value;
        }
        if other.min_residual_seen < self.min_residual_seen {
            self.min_residual_seen = other.min_residual_seen;
        }
    }
}

#[inline]
fn clamp_min(value: f64, min_value: f64) -> f64 {
    if value < min_value {
        min_value
    } else {
        value
    }
}

pub fn guarded_pow(base: f64, exp: f64, max_log: f64) -> f64 {
    guarded_pow_with_stats(base, exp, max_log, None)
}

pub fn guarded_pow_with_stats(
    base: f64,
    exp: f64,
    max_log: f64,
    mut stats: Option<&mut BarrierClampStats>,
) -> f64 {
    if base <= 0.0 {
        if let Some(stats) = stats.as_mut() {
            stats.clamped_powers += 1;
        }
        return 0.0;
    }
    let log_base = base.ln();
    let clamped_log = log_base.clamp(-max_log, max_log);
    if clamped_log != log_base {
        if let Some(stats) = stats.as_mut() {
            stats.clamped_logs += 1;
        }
    }
    let log_val = exp * clamped_log;
    if log_val > max_log {
        if let Some(stats) = stats.as_mut() {
            stats.clamped_powers += 1;
        }
        f64::INFINITY
    } else if log_val < -max_log {
        if let Some(stats) = stats.as_mut() {
            stats.clamped_powers += 1;
        }
        0.0
    } else {
        log_val.exp()
    }
}

pub fn guarded_log(x: f64, min_x: f64) -> f64 {
    guarded_log_with_stats(x, min_x, None)
}

pub fn guarded_log_with_stats(
    x: f64,
    min_x: f64,
    mut stats: Option<&mut BarrierClampStats>,
) -> f64 {
    if x <= 0.0 {
        if let Some(stats) = stats.as_mut() {
            stats.clamped_logs += 1;
        }
        return f64::NEG_INFINITY;
    }
    let clamped = x.max(min_x);
    if clamped != x {
        if let Some(stats) = stats.as_mut() {
            stats.clamped_logs += 1;
        }
    }
    clamped.ln()
}

pub fn guarded_inverse(x: f64, min_x: f64) -> f64 {
    guarded_inverse_with_stats(x, min_x, None)
}

pub fn guarded_inverse_with_stats(
    x: f64,
    min_x: f64,
    mut stats: Option<&mut BarrierClampStats>,
) -> f64 {
    if x.abs() < min_x {
        if let Some(stats) = stats.as_mut() {
            stats.clamped_inverses += 1;
        }
        return 1.0 / min_x.copysign(x);
    }
    1.0 / x
}

pub fn guarded_pow_slice(values: &[f64], exp: f64, max_log: f64) -> Vec<f64> {
    values
        .iter()
        .map(|&value| guarded_pow(value, exp, max_log))
        .collect()
}

pub fn guarded_log_slice(values: &[f64], min_x: f64) -> Vec<f64> {
    values
        .iter()
        .map(|&value| guarded_log(value, min_x))
        .collect()
}

pub fn guarded_inverse_slice(values: &[f64], min_x: f64) -> Vec<f64> {
    values
        .iter()
        .map(|&value| guarded_inverse(value, min_x))
        .collect()
}

pub fn clamped_alpha(alpha: f64, config: &BarrierClampConfig) -> f64 {
    clamped_alpha_with_stats(alpha, config, None)
}

pub fn clamped_alpha_with_stats(
    alpha: f64,
    config: &BarrierClampConfig,
    mut stats: Option<&mut BarrierClampStats>,
) -> f64 {
    let clamped = alpha.clamp(config.alpha_min, config.alpha_max);
    if clamped != alpha {
        if let Some(stats) = stats.as_mut() {
            stats.clamped_alpha += 1;
        }
    }
    clamped
}

pub fn clamped_barrier_term(
    residual: f64,
    alpha: f64,
    config: &BarrierClampConfig,
    mut stats: Option<&mut BarrierClampStats>,
) -> f64 {
    if residual <= 0.0 {
        if let Some(stats) = stats.as_mut() {
            stats.clamped_residuals += 1;
            stats.clamped_barrier += 1;
        }
        return config.barrier_clamp_max;
    }
    let clamped_residual = residual.max(config.residual_min);
    if clamped_residual != residual {
        if let Some(stats) = stats.as_mut() {
            stats.clamped_residuals += 1;
        }
    }
    let powered = guarded_pow_with_stats(
        clamped_residual,
        -alpha,
        config.max_log,
        stats.as_deref_mut(),
    );
    let clamped = powered.clamp(0.0, config.barrier_clamp_max);
    if clamped != powered {
        if let Some(stats) = stats.as_mut() {
            stats.clamped_barrier += 1;
        }
    }
    if let Some(stats) = stats.as_mut() {
        stats.record_barrier_value(clamped);
        stats.record_residual(clamped_residual);
    }
    clamped
}

pub fn clamped_gradient_term(
    residual: f64,
    alpha: f64,
    config: &BarrierClampConfig,
    mut stats: Option<&mut BarrierClampStats>,
) -> f64 {
    if residual <= config.residual_min {
        if let Some(stats) = stats.as_mut() {
            stats.clamped_residuals += 1;
            stats.clamped_gradient += 1;
        }
        return config.gradient_clamp_max;
    }
    let powered =
        guarded_pow_with_stats(residual, -alpha - 1.0, config.max_log, stats.as_deref_mut());
    let deriv = alpha * powered;
    let clamped = deriv.clamp(-config.gradient_clamp_max, config.gradient_clamp_max);
    if clamped != deriv {
        if let Some(stats) = stats.as_mut() {
            stats.clamped_gradient += 1;
        }
    }
    if let Some(stats) = stats.as_mut() {
        stats.record_residual(residual);
    }
    clamped
}

pub fn barrier_inverse_power_simd(x: &[f64], alpha: f64) -> Vec<f64> {
    x.iter().map(|&value| value.powf(-alpha)).collect()
}

pub fn barrier_power_simd(x: &[f64], alpha: f64) -> Vec<f64> {
    x.iter().map(|&value| value.powf(alpha)).collect()
}

fn preprocess_deltas_simd(
    flow: &[f64],
    lower: &[f64],
    upper: &[f64],
    min_value: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut upper_delta = vec![0.0; flow.len()];
    let mut lower_delta = vec![0.0; flow.len()];
    for idx in 0..flow.len() {
        upper_delta[idx] = clamp_min(upper[idx] - flow[idx], min_value);
        lower_delta[idx] = clamp_min(flow[idx] - lower[idx], min_value);
    }
    (upper_delta, lower_delta)
}

fn barrier_term(delta: f64, beta: f64) -> f64 {
    let config = BarrierClampConfig::default();
    clamped_barrier_term(delta, beta, &config, None)
}

fn barrier_term_derivative(delta: f64, beta: f64) -> f64 {
    let config = BarrierClampConfig::default();
    -clamped_gradient_term(delta, beta, &config, None)
}

fn barrier_lengths_simd_slice(
    upper_delta: &[f64],
    lower_delta: &[f64],
    beta: f64,
    output: &mut [f64],
) {
    for (out, (upper_d, lower_d)) in output
        .iter_mut()
        .zip(upper_delta.iter().zip(lower_delta.iter()))
    {
        *out = barrier_term(*upper_d, beta) + barrier_term(*lower_d, beta);
    }
}

fn barrier_gradient_simd_slice(
    upper_delta: &[f64],
    lower_delta: &[f64],
    beta: f64,
    output: &mut [f64],
) {
    for (out, (upper_d, lower_d)) in output
        .iter_mut()
        .zip(upper_delta.iter().zip(lower_delta.iter()))
    {
        let upper_term = -barrier_term_derivative(*upper_d, beta);
        let lower_term = barrier_term_derivative(*lower_d, beta);
        *out = upper_term + lower_term;
    }
}

pub fn safe_log(value: f64, min_value: f64) -> f64 {
    value.max(min_value).ln()
}

pub fn safe_exp(value: f64, max_exponent: f64) -> f64 {
    value.min(max_exponent).exp()
}

pub fn barrier_lengths(
    flow: &[f64],
    lower: &[f64],
    upper: &[f64],
    beta: f64,
    min_value: f64,
    threads: usize,
) -> Vec<f64> {
    assert_eq!(flow.len(), lower.len());
    assert_eq!(flow.len(), upper.len());

    let (upper_delta, lower_delta) = preprocess_deltas_simd(flow, lower, upper, min_value);

    #[cfg(feature = "parallel")]
    {
        let mut output = vec![0.0; upper_delta.len()];
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        let threads = if threads == 0 {
            available
        } else {
            threads.min(available)
        }
        .max(1);
        let chunk_size = upper_delta.len().div_ceil(threads);
        std::thread::scope(|scope| {
            for (chunk_index, out_chunk) in output.chunks_mut(chunk_size).enumerate() {
                let start = chunk_index * chunk_size;
                let end = start + out_chunk.len();
                let upper_chunk = &upper_delta[start..end];
                let lower_chunk = &lower_delta[start..end];
                scope.spawn(move || {
                    barrier_lengths_simd_slice(upper_chunk, lower_chunk, beta, out_chunk);
                });
            }
        });
        output
    }

    #[cfg(not(feature = "parallel"))]
    {
        let _ = threads;
        let mut output = vec![0.0; upper_delta.len()];
        barrier_lengths_simd_slice(&upper_delta, &lower_delta, beta, &mut output);
        output
    }
}

pub fn barrier_gradient(
    flow: &[f64],
    lower: &[f64],
    upper: &[f64],
    beta: f64,
    min_value: f64,
    threads: usize,
) -> Vec<f64> {
    assert_eq!(flow.len(), lower.len());
    assert_eq!(flow.len(), upper.len());

    let (upper_delta, lower_delta) = preprocess_deltas_simd(flow, lower, upper, min_value);

    #[cfg(feature = "parallel")]
    {
        let mut output = vec![0.0; upper_delta.len()];
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        let threads = if threads == 0 {
            available
        } else {
            threads.min(available)
        }
        .max(1);
        let chunk_size = upper_delta.len().div_ceil(threads);
        std::thread::scope(|scope| {
            for (chunk_index, out_chunk) in output.chunks_mut(chunk_size).enumerate() {
                let start = chunk_index * chunk_size;
                let end = start + out_chunk.len();
                let upper_chunk = &upper_delta[start..end];
                let lower_chunk = &lower_delta[start..end];
                scope.spawn(move || {
                    barrier_gradient_simd_slice(upper_chunk, lower_chunk, beta, out_chunk);
                });
            }
        });
        output
    }

    #[cfg(not(feature = "parallel"))]
    {
        let _ = threads;
        let mut output = vec![0.0; upper_delta.len()];
        barrier_gradient_simd_slice(&upper_delta, &lower_delta, beta, &mut output);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn assert_close(lhs: f64, rhs: f64, epsilon: f64) {
        let diff = (lhs - rhs).abs();
        assert!(
            diff <= epsilon,
            "expected {lhs} and {rhs} to be within {epsilon}, diff={diff}"
        );
    }

    fn serial_lengths(
        flow: &[f64],
        lower: &[f64],
        upper: &[f64],
        beta: f64,
        min_value: f64,
    ) -> Vec<f64> {
        flow.iter()
            .zip(lower.iter())
            .zip(upper.iter())
            .map(|((f, l), u)| {
                let upper_delta = clamp_min(u - f, min_value);
                let lower_delta = clamp_min(f - l, min_value);
                barrier_term(upper_delta, beta) + barrier_term(lower_delta, beta)
            })
            .collect()
    }

    fn serial_gradient(
        flow: &[f64],
        lower: &[f64],
        upper: &[f64],
        beta: f64,
        min_value: f64,
    ) -> Vec<f64> {
        flow.iter()
            .zip(lower.iter())
            .zip(upper.iter())
            .map(|((f, l), u)| {
                let upper_delta = clamp_min(u - f, min_value);
                let lower_delta = clamp_min(f - l, min_value);
                let upper_term = -barrier_term_derivative(upper_delta, beta);
                let lower_term = barrier_term_derivative(lower_delta, beta);
                upper_term + lower_term
            })
            .collect()
    }

    /// Helper to compute a scalar "barrier energy" that mirrors a common usage
    /// pattern in optimization: sum the per-variable penalty terms.
    fn barrier_energy(
        flow: &[f64],
        lower: &[f64],
        upper: &[f64],
        beta: f64,
        min_value: f64,
    ) -> f64 {
        barrier_lengths(flow, lower, upper, beta, min_value, 0)
            .iter()
            .sum::<f64>()
    }

    #[test]
    fn lengths_match_serial() {
        let flow = vec![2.0, 4.5, 7.2, 1.2, 9.1];
        let lower = vec![0.0, 1.0, 2.0, 0.0, 5.0];
        let upper = vec![10.0, 10.0, 10.0, 3.0, 12.0];
        let beta = 0.5;
        let min_value = 1e-9;

        let expected = serial_lengths(&flow, &lower, &upper, beta, min_value);
        let actual = barrier_lengths(&flow, &lower, &upper, beta, min_value, 0);

        for (a, b) in expected.iter().zip(actual.iter()) {
            assert_close(*a, *b, 1e-12);
        }
    }

    #[test]
    fn gradient_match_serial() {
        let flow = vec![2.0, 4.5, 7.2, 1.2, 9.1];
        let lower = vec![0.0, 1.0, 2.0, 0.0, 5.0];
        let upper = vec![10.0, 10.0, 10.0, 3.0, 12.0];
        let beta = 0.5;
        let min_value = 1e-9;

        let expected = serial_gradient(&flow, &lower, &upper, beta, min_value);
        let actual = barrier_gradient(&flow, &lower, &upper, beta, min_value, 0);

        for (a, b) in expected.iter().zip(actual.iter()) {
            assert_close(*a, *b, 1e-12);
        }
    }

    #[test]
    fn clamps_near_bounds() {
        let flow = vec![1e-12, 9.999999999, 5.0];
        let lower = vec![0.0, 0.0, 5.0];
        let upper = vec![10.0, 10.0, 10.0];
        let beta = 0.5;
        let min_value = 1e-9;

        let lengths = barrier_lengths(&flow, &lower, &upper, beta, min_value, 0);
        let gradient = barrier_gradient(&flow, &lower, &upper, beta, min_value, 0);

        for value in lengths.iter().chain(gradient.iter()) {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        // This test demonstrates a practical use-case: verify that the analytic
        // gradient matches a numerical approximation for use in gradient-based
        // optimizers (e.g., projected gradient descent).
        let mut flow = vec![2.5, 4.0, 6.0];
        let lower = vec![1.0, 2.0, 4.0];
        let upper = vec![10.0, 8.0, 9.0];
        let beta = 0.5;
        let min_value = 1e-9;
        let eps = 1e-6;

        let analytic = barrier_gradient(&flow, &lower, &upper, beta, min_value, 0);

        for idx in 0..flow.len() {
            let original = flow[idx];
            flow[idx] = original + eps;
            let plus = barrier_energy(&flow, &lower, &upper, beta, min_value);
            flow[idx] = original - eps;
            let minus = barrier_energy(&flow, &lower, &upper, beta, min_value);
            flow[idx] = original;

            let numerical = (plus - minus) / (2.0 * eps);
            assert_close(analytic[idx], numerical, 1e-6);
        }
    }

    #[test]
    fn barrier_penalty_increases_near_bounds() {
        // Another usage pattern: check that constraints are enforced by a
        // rapidly increasing penalty near bounds.
        let lower = vec![0.0];
        let upper = vec![10.0];
        let beta = 0.5;
        let min_value = 1e-9;

        let middle = barrier_lengths(&[5.0], &lower, &upper, beta, min_value, 0)[0];
        let near_lower = barrier_lengths(&[0.1], &lower, &upper, beta, min_value, 0)[0];
        let near_upper = barrier_lengths(&[9.9], &lower, &upper, beta, min_value, 0)[0];

        assert!(near_lower > middle);
        assert!(near_upper > middle);
    }

    #[test]
    fn symmetric_bounds_have_zero_gradient_at_midpoint() {
        // For symmetric bounds, the midpoint should yield zero gradient,
        // illustrating how the barrier can stabilize around feasible centers.
        let flow = vec![0.0];
        let lower = vec![-5.0];
        let upper = vec![5.0];
        let beta = 0.5;
        let min_value = 1e-9;

        let gradient = barrier_gradient(&flow, &lower, &upper, beta, min_value, 0);
        assert_close(gradient[0], 0.0, 1e-12);
    }

    #[test]
    fn matches_power_barrier_reference() {
        // Compare against the closed-form "power barrier" (delta^-beta) to show
        // equivalence with a well-known alternative formulation.
        let flow = vec![3.0, 4.0];
        let lower = vec![0.0, 1.0];
        let upper = vec![10.0, 9.0];
        let beta = 0.5;
        let min_value = 1e-9;

        let lengths = barrier_lengths(&flow, &lower, &upper, beta, min_value, 0);
        let reference: Vec<f64> = flow
            .iter()
            .zip(lower.iter())
            .zip(upper.iter())
            .map(|((f, l), u)| {
                let upper_delta = clamp_min(u - f, min_value);
                let lower_delta = clamp_min(f - l, min_value);
                upper_delta.powf(-beta) + lower_delta.powf(-beta)
            })
            .collect();

        for (a, b) in lengths.iter().zip(reference.iter()) {
            assert_close(*a, *b, 1e-12);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_lengths_match_serial() {
        let flow = vec![2.0, 4.5, 7.2, 1.2, 9.1, 3.3, 6.7, 8.8];
        let lower = vec![0.0, 1.0, 2.0, 0.0, 5.0, 0.5, 1.1, 2.2];
        let upper = vec![10.0, 10.0, 10.0, 3.0, 12.0, 9.5, 8.0, 11.0];
        let beta = 0.5;
        let min_value = 1e-9;

        let expected = serial_lengths(&flow, &lower, &upper, beta, min_value);
        let actual = barrier_lengths(&flow, &lower, &upper, beta, min_value, 0);

        for (a, b) in expected.iter().zip(actual.iter()) {
            assert_close(*a, *b, 1e-12);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_gradient_match_serial() {
        let flow = vec![2.0, 4.5, 7.2, 1.2, 9.1, 3.3, 6.7, 8.8];
        let lower = vec![0.0, 1.0, 2.0, 0.0, 5.0, 0.5, 1.1, 2.2];
        let upper = vec![10.0, 10.0, 10.0, 3.0, 12.0, 9.5, 8.0, 11.0];
        let beta = 0.5;
        let min_value = 1e-9;

        let expected = serial_gradient(&flow, &lower, &upper, beta, min_value);
        let actual = barrier_gradient(&flow, &lower, &upper, beta, min_value, 0);

        for (a, b) in expected.iter().zip(actual.iter()) {
            assert_close(*a, *b, 1e-12);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_preprocess_matches_scalar() {
        let flow = vec![2.0, 4.5, 7.2, 1.2, 9.1, 3.3, 6.7];
        let lower = vec![0.0, 1.0, 2.0, 0.0, 5.0, 0.5, 1.1];
        let upper = vec![10.0, 10.0, 10.0, 3.0, 12.0, 9.5, 8.0];
        let min_value = 1e-9;

        let (upper_delta, lower_delta) = preprocess_deltas_simd(&flow, &lower, &upper, min_value);
        for idx in 0..flow.len() {
            let expected_upper = clamp_min(upper[idx] - flow[idx], min_value);
            let expected_lower = clamp_min(flow[idx] - lower[idx], min_value);
            assert_close(upper_delta[idx], expected_upper, 1e-12);
            assert_close(lower_delta[idx], expected_lower, 1e-12);
        }
    }
}
