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

    let mut step = max_step * 0.99;
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

    let mut step = max_step * 0.99;
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
