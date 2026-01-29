#[inline]
fn clamp_min(value: f64, min_value: f64) -> f64 {
    if value < min_value {
        min_value
    } else {
        value
    }
}

#[cfg(feature = "simd")]
fn preprocess_deltas_simd(
    flow: &[f64],
    lower: &[f64],
    upper: &[f64],
    min_value: f64,
) -> (Vec<f64>, Vec<f64>) {
    use std::simd::{Simd, SimdFloat};

    let lanes = 4;
    let mut upper_delta = vec![0.0; flow.len()];
    let mut lower_delta = vec![0.0; flow.len()];
    let min_vec = Simd::splat(min_value);

    let chunks = flow.len() / lanes;
    for i in 0..chunks {
        let base = i * lanes;
        let flow_v = Simd::from_slice(&flow[base..base + lanes]);
        let upper_v = Simd::from_slice(&upper[base..base + lanes]);
        let lower_v = Simd::from_slice(&lower[base..base + lanes]);

        let upper_delta_v = (upper_v - flow_v).simd_max(min_vec);
        let lower_delta_v = (flow_v - lower_v).simd_max(min_vec);

        upper_delta_v.write_to_slice(&mut upper_delta[base..base + lanes]);
        lower_delta_v.write_to_slice(&mut lower_delta[base..base + lanes]);
    }

    for idx in (chunks * lanes)..flow.len() {
        upper_delta[idx] = clamp_min(upper[idx] - flow[idx], min_value);
        lower_delta[idx] = clamp_min(flow[idx] - lower[idx], min_value);
    }

    (upper_delta, lower_delta)
}

#[cfg(not(feature = "simd"))]
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

fn barrier_term(delta: f64, alpha: f64) -> f64 {
    (-((1.0 + alpha) * delta.ln())).exp()
}

fn barrier_term_derivative(delta: f64, alpha: f64) -> f64 {
    -((1.0 + alpha) / delta) * barrier_term(delta, alpha)
}

pub fn barrier_lengths(
    flow: &[f64],
    lower: &[f64],
    upper: &[f64],
    alpha: f64,
    min_value: f64,
) -> Vec<f64> {
    assert_eq!(flow.len(), lower.len());
    assert_eq!(flow.len(), upper.len());

    let (upper_delta, lower_delta) = preprocess_deltas_simd(flow, lower, upper, min_value);

    #[cfg(feature = "parallel")]
    {
        let mut output = vec![0.0; upper_delta.len()];
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let chunk_size = (upper_delta.len() + threads - 1) / threads;
        std::thread::scope(|scope| {
            for (chunk_index, out_chunk) in output.chunks_mut(chunk_size).enumerate() {
                let start = chunk_index * chunk_size;
                let end = start + out_chunk.len();
                let upper_chunk = &upper_delta[start..end];
                let lower_chunk = &lower_delta[start..end];
                scope.spawn(move || {
                    for ((out, upper_d), lower_d) in out_chunk
                        .iter_mut()
                        .zip(upper_chunk.iter())
                        .zip(lower_chunk.iter())
                    {
                        *out = barrier_term(*upper_d, alpha) + barrier_term(*lower_d, alpha);
                    }
                });
            }
        });
        output
    }

    #[cfg(not(feature = "parallel"))]
    {
        upper_delta
            .iter()
            .zip(lower_delta.iter())
            .map(|(upper_d, lower_d)| barrier_term(*upper_d, alpha) + barrier_term(*lower_d, alpha))
            .collect()
    }
}

pub fn barrier_gradient(
    flow: &[f64],
    lower: &[f64],
    upper: &[f64],
    alpha: f64,
    min_value: f64,
) -> Vec<f64> {
    assert_eq!(flow.len(), lower.len());
    assert_eq!(flow.len(), upper.len());

    let (upper_delta, lower_delta) = preprocess_deltas_simd(flow, lower, upper, min_value);

    #[cfg(feature = "parallel")]
    {
        let mut output = vec![0.0; upper_delta.len()];
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let chunk_size = (upper_delta.len() + threads - 1) / threads;
        std::thread::scope(|scope| {
            for (chunk_index, out_chunk) in output.chunks_mut(chunk_size).enumerate() {
                let start = chunk_index * chunk_size;
                let end = start + out_chunk.len();
                let upper_chunk = &upper_delta[start..end];
                let lower_chunk = &lower_delta[start..end];
                scope.spawn(move || {
                    for ((out, upper_d), lower_d) in out_chunk
                        .iter_mut()
                        .zip(upper_chunk.iter())
                        .zip(lower_chunk.iter())
                    {
                        let upper_term = -barrier_term_derivative(*upper_d, alpha);
                        let lower_term = barrier_term_derivative(*lower_d, alpha);
                        *out = upper_term + lower_term;
                    }
                });
            }
        });
        output
    }

    #[cfg(not(feature = "parallel"))]
    {
        upper_delta
            .iter()
            .zip(lower_delta.iter())
            .map(|(upper_d, lower_d)| {
                let upper_term = -barrier_term_derivative(*upper_d, alpha);
                let lower_term = barrier_term_derivative(*lower_d, alpha);
                upper_term + lower_term
            })
            .collect()
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
        alpha: f64,
        min_value: f64,
    ) -> Vec<f64> {
        flow.iter()
            .zip(lower.iter())
            .zip(upper.iter())
            .map(|((f, l), u)| {
                let upper_delta = clamp_min(u - f, min_value);
                let lower_delta = clamp_min(f - l, min_value);
                barrier_term(upper_delta, alpha) + barrier_term(lower_delta, alpha)
            })
            .collect()
    }

    fn serial_gradient(
        flow: &[f64],
        lower: &[f64],
        upper: &[f64],
        alpha: f64,
        min_value: f64,
    ) -> Vec<f64> {
        flow.iter()
            .zip(lower.iter())
            .zip(upper.iter())
            .map(|((f, l), u)| {
                let upper_delta = clamp_min(u - f, min_value);
                let lower_delta = clamp_min(f - l, min_value);
                let upper_term = -barrier_term_derivative(upper_delta, alpha);
                let lower_term = barrier_term_derivative(lower_delta, alpha);
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
        alpha: f64,
        min_value: f64,
    ) -> f64 {
        barrier_lengths(flow, lower, upper, alpha, min_value)
            .iter()
            .sum::<f64>()
    }

    #[test]
    fn lengths_match_serial() {
        let flow = vec![2.0, 4.5, 7.2, 1.2, 9.1];
        let lower = vec![0.0, 1.0, 2.0, 0.0, 5.0];
        let upper = vec![10.0, 10.0, 10.0, 3.0, 12.0];
        let alpha = 0.01;
        let min_value = 1e-9;

        let expected = serial_lengths(&flow, &lower, &upper, alpha, min_value);
        let actual = barrier_lengths(&flow, &lower, &upper, alpha, min_value);

        for (a, b) in expected.iter().zip(actual.iter()) {
            assert_close(*a, *b, 1e-12);
        }
    }

    #[test]
    fn gradient_match_serial() {
        let flow = vec![2.0, 4.5, 7.2, 1.2, 9.1];
        let lower = vec![0.0, 1.0, 2.0, 0.0, 5.0];
        let upper = vec![10.0, 10.0, 10.0, 3.0, 12.0];
        let alpha = 0.02;
        let min_value = 1e-9;

        let expected = serial_gradient(&flow, &lower, &upper, alpha, min_value);
        let actual = barrier_gradient(&flow, &lower, &upper, alpha, min_value);

        for (a, b) in expected.iter().zip(actual.iter()) {
            assert_close(*a, *b, 1e-12);
        }
    }

    #[test]
    fn clamps_near_bounds() {
        let flow = vec![1e-12, 9.999999999, 5.0];
        let lower = vec![0.0, 0.0, 5.0];
        let upper = vec![10.0, 10.0, 10.0];
        let alpha = 0.03;
        let min_value = 1e-9;

        let lengths = barrier_lengths(&flow, &lower, &upper, alpha, min_value);
        let gradient = barrier_gradient(&flow, &lower, &upper, alpha, min_value);

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
        let alpha = 0.01;
        let min_value = 1e-9;
        let eps = 1e-6;

        let analytic = barrier_gradient(&flow, &lower, &upper, alpha, min_value);

        for idx in 0..flow.len() {
            let original = flow[idx];
            flow[idx] = original + eps;
            let plus = barrier_energy(&flow, &lower, &upper, alpha, min_value);
            flow[idx] = original - eps;
            let minus = barrier_energy(&flow, &lower, &upper, alpha, min_value);
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
        let alpha = 0.05;
        let min_value = 1e-9;

        let middle = barrier_lengths(&[5.0], &lower, &upper, alpha, min_value)[0];
        let near_lower = barrier_lengths(&[0.1], &lower, &upper, alpha, min_value)[0];
        let near_upper = barrier_lengths(&[9.9], &lower, &upper, alpha, min_value)[0];

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
        let alpha = 0.1;
        let min_value = 1e-9;

        let gradient = barrier_gradient(&flow, &lower, &upper, alpha, min_value);
        assert_close(gradient[0], 0.0, 1e-12);
    }

    #[test]
    fn matches_power_barrier_reference() {
        // Compare against the closed-form "power barrier" (delta^-(1+alpha))
        // to show equivalence with a well-known alternative formulation.
        let flow = vec![3.0, 4.0];
        let lower = vec![0.0, 1.0];
        let upper = vec![10.0, 9.0];
        let alpha = 0.2;
        let min_value = 1e-9;

        let lengths = barrier_lengths(&flow, &lower, &upper, alpha, min_value);
        let reference: Vec<f64> = flow
            .iter()
            .zip(lower.iter())
            .zip(upper.iter())
            .map(|((f, l), u)| {
                let upper_delta = clamp_min(u - f, min_value);
                let lower_delta = clamp_min(f - l, min_value);
                upper_delta.powf(-(1.0 + alpha)) + lower_delta.powf(-(1.0 + alpha))
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
        let alpha = 0.01;
        let min_value = 1e-9;

        let expected = serial_lengths(&flow, &lower, &upper, alpha, min_value);
        let actual = barrier_lengths(&flow, &lower, &upper, alpha, min_value);

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
        let alpha = 0.01;
        let min_value = 1e-9;

        let expected = serial_gradient(&flow, &lower, &upper, alpha, min_value);
        let actual = barrier_gradient(&flow, &lower, &upper, alpha, min_value);

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
