use super::Potential;

pub struct LineSearchInput<'a> {
    pub flow: &'a [f64],
    pub delta: &'a [f64],
    pub cost: &'a [f64],
    pub lower: &'a [f64],
    pub upper: &'a [f64],
    pub potential: &'a Potential,
    pub current_potential: f64,
    pub required_reduction: f64,
    pub gradient: &'a [f64],
    pub lengths: &'a [f64],
}

pub fn line_search(input: LineSearchInput<'_>) -> Option<(Vec<f64>, f64)> {
    let LineSearchInput {
        flow,
        delta,
        cost,
        lower,
        upper,
        potential,
        current_potential,
        required_reduction,
        gradient,
        lengths,
    } = input;
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

    let mut step = taylor_step(delta, gradient, lengths, max_step);
    let reduction_floor = potential.reduction_floor(current_potential);
    for _ in 0..20 {
        let candidate_flow: Vec<f64> = flow
            .iter()
            .zip(delta.iter())
            .map(|(f, d)| f + step * d)
            .collect();
        let candidate_potential = potential.value(cost, &candidate_flow, lower, upper);
        if candidate_potential <= current_potential - required_reduction {
            return Some((candidate_flow, step));
        }
        step *= 0.5;
    }

    let mut step = taylor_step(delta, gradient, lengths, max_step);
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

fn taylor_step(delta: &[f64], gradient: &[f64], lengths: &[f64], max_step: f64) -> f64 {
    let mut linear = 0.0;
    let mut quadratic = 0.0;
    for ((d, g), l) in delta.iter().zip(gradient.iter()).zip(lengths.iter()) {
        linear += g * d;
        quadratic += l * d * d;
    }
    if quadratic <= 0.0 {
        return max_step * 0.5;
    }
    let step = (-linear / quadratic).clamp(0.0, max_step * 0.99);
    if step <= 0.0 {
        max_step * 0.5
    } else {
        step
    }
}
