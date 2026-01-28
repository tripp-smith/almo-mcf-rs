use rayon::prelude::*;

#[inline]
fn clamp_min(value: f64, min_value: f64) -> f64 {
    if value < min_value {
        min_value
    } else {
        value
    }
}

#[cfg(feature = "simd")]
fn preprocess_deltas_simd(flow: &[f64], lower: &[f64], upper: &[f64], min_value: f64) -> (Vec<f64>, Vec<f64>) {
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
fn preprocess_deltas_simd(flow: &[f64], lower: &[f64], upper: &[f64], min_value: f64) -> (Vec<f64>, Vec<f64>) {
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

pub fn barrier_lengths(flow: &[f64], lower: &[f64], upper: &[f64], alpha: f64, min_value: f64) -> Vec<f64> {
    assert_eq!(flow.len(), lower.len());
    assert_eq!(flow.len(), upper.len());

    let (upper_delta, lower_delta) = preprocess_deltas_simd(flow, lower, upper, min_value);

    #[cfg(feature = "parallel")]
    {
        upper_delta
            .par_iter()
            .zip(lower_delta.par_iter())
            .map(|(upper_d, lower_d)| barrier_term(*upper_d, alpha) + barrier_term(*lower_d, alpha))
            .collect()
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

pub fn barrier_gradient(flow: &[f64], lower: &[f64], upper: &[f64], alpha: f64, min_value: f64) -> Vec<f64> {
    assert_eq!(flow.len(), lower.len());
    assert_eq!(flow.len(), upper.len());

    let (upper_delta, lower_delta) = preprocess_deltas_simd(flow, lower, upper, min_value);

    #[cfg(feature = "parallel")]
    {
        upper_delta
            .par_iter()
            .zip(lower_delta.par_iter())
            .map(|(upper_d, lower_d)| {
                let upper_term = -barrier_term_derivative(*upper_d, alpha);
                let lower_term = barrier_term_derivative(*lower_d, alpha);
                upper_term + lower_term
            })
            .collect()
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
    use approx::assert_relative_eq;

    fn serial_lengths(flow: &[f64], lower: &[f64], upper: &[f64], alpha: f64, min_value: f64) -> Vec<f64> {
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

    fn serial_gradient(flow: &[f64], lower: &[f64], upper: &[f64], alpha: f64, min_value: f64) -> Vec<f64> {
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
            assert_relative_eq!(a, b, epsilon = 1e-12);
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
            assert_relative_eq!(a, b, epsilon = 1e-12);
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
}
