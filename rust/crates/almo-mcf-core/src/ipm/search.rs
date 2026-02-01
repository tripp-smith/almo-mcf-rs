use super::Potential;

pub fn line_search(
    flow: &[f64],
    delta: &[f64],
    cost: &[f64],
    lower: &[f64],
    upper: &[f64],
    potential: &Potential,
    current_potential: f64,
) -> Option<(Vec<f64>, f64)> {
    let mut max_step = f64::INFINITY;
    for (idx, &d) in delta.iter().enumerate() {
        if d > 0.0 {
            let slack = upper[idx] - flow[idx];
            max_step = max_step.min(slack / d);
        } else if d < 0.0 {
            let slack = flow[idx] - lower[idx];
            max_step = max_step.min(slack / -d);
        }
    }

    if !max_step.is_finite() || max_step <= 0.0 {
        return None;
    }

    let mut step = max_step * 0.99;
    let reduction_floor = potential.reduction_floor(current_potential);
    for _ in 0..20 {
        let candidate_flow: Vec<f64> = flow
            .iter()
            .zip(delta.iter())
            .map(|(f, d)| f + step * d)
            .collect();
        let candidate_potential = potential.value(cost, &candidate_flow, lower, upper);
        if candidate_potential <= current_potential - reduction_floor {
            return Some((candidate_flow, step));
        }
        step *= 0.5;
    }

    None
}
