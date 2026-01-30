use crate::numerics::{EPSILON, GRADIENT_EPSILON};

pub fn gradient_proxy(
    cost: &[f64],
    flow: &[f64],
    lower: &[f64],
    upper: &[f64],
    mu: f64,
    epsilon: f64,
) -> Vec<f64> {
    assert_eq!(cost.len(), flow.len());
    assert_eq!(lower.len(), flow.len());
    assert_eq!(upper.len(), flow.len());
    let min_value = if epsilon > 0.0 { epsilon } else { EPSILON };
    cost.iter()
        .zip(flow.iter())
        .zip(lower.iter().zip(upper.iter()))
        .map(|((c, f), (l, u))| {
            let upper_delta = (u - f).max(min_value);
            let lower_delta = (f - l).max(min_value);
            c + mu * (1.0 / upper_delta - 1.0 / lower_delta)
        })
        .collect()
}

pub fn duality_gap_proxy(gradient: &[f64], flow: &[f64]) -> f64 {
    assert_eq!(gradient.len(), flow.len());
    gradient
        .iter()
        .zip(flow.iter())
        .map(|(g, f)| g.abs() * (f.abs() + GRADIENT_EPSILON))
        .sum()
}
