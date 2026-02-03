use crate::{McfError, McfProblem};
use num_bigint::BigInt;
use num_traits::{Signed, Zero};

pub fn flow_to_i64(flow: &[f64]) -> Vec<i64> {
    flow.iter().map(|value| value.round() as i64).collect()
}

pub fn scale_flow(flow: &[f64], factor: i64) -> Vec<i64> {
    flow.iter()
        .map(|value| {
            (value.round() as i128)
                .checked_mul(factor as i128)
                .map(|value| value as i64)
                .unwrap_or_else(|| {
                    if value.is_sign_negative() {
                        i64::MIN
                    } else {
                        i64::MAX
                    }
                })
        })
        .collect()
}

pub fn scale_problem(problem: &McfProblem, divisor: i64) -> Result<McfProblem, McfError> {
    if divisor <= 0 {
        return Err(McfError::InvalidInput(
            "capacity scaling divisor must be positive".to_string(),
        ));
    }

    let lower = scale_values(&problem.lower, divisor)?;
    let upper = scale_values(&problem.upper, divisor)?;
    let demands = scale_values(&problem.demands, divisor)?;

    McfProblem::new(
        problem.tails.clone(),
        problem.heads.clone(),
        lower,
        upper,
        problem.cost.clone(),
        demands,
    )
}

pub fn scale_values(values: &[i64], divisor: i64) -> Result<Vec<i64>, McfError> {
    values
        .iter()
        .map(|&value| {
            if value % divisor != 0 {
                return Err(McfError::InvalidInput(
                    "capacity scaling requires divisible values".to_string(),
                ));
            }
            Ok(value / divisor)
        })
        .collect()
}

pub fn max_power_of_two_divisor(lower: &[i64], upper: &[i64], demands: &[i64]) -> i64 {
    let mut min_trailing = u32::MAX;
    for value in lower.iter().chain(upper.iter()).chain(demands.iter()) {
        if *value == 0 {
            continue;
        }
        let trailing = value.abs().trailing_zeros();
        min_trailing = min_trailing.min(trailing);
    }
    if min_trailing == u32::MAX {
        return 1;
    }
    1_i64 << min_trailing.min(62)
}

pub fn max_abs(values: &[i64]) -> i64 {
    values.iter().map(|value| value.abs()).max().unwrap_or(0)
}

pub fn next_power_of_two(value: i64) -> i64 {
    if value <= 1 {
        1
    } else {
        let shifted = (value - 1) as u64;
        shifted.next_power_of_two() as i64
    }
}

pub fn bigint_bits(value: &BigInt) -> u64 {
    if value.is_zero() {
        return 1;
    }
    let mut value = value.abs();
    let mut bits = 0;
    while value > BigInt::zero() {
        value >>= 1;
        bits += 1;
    }
    bits.max(1)
}

pub fn feasible_initial_flow(problem: &McfProblem, flow: Option<Vec<i64>>) -> Option<Vec<i64>> {
    let flow = flow?;
    if flow.len() != problem.edge_count() {
        return None;
    }
    let mut balance = vec![0_i64; problem.node_count];
    for (idx, &value) in flow.iter().enumerate() {
        let lower = problem.lower[idx];
        let upper = problem.upper[idx];
        if value < lower || value > upper {
            return None;
        }
        let tail = problem.tails[idx] as usize;
        let head = problem.heads[idx] as usize;
        balance[tail] = balance[tail].checked_sub(value)?;
        balance[head] = balance[head].checked_add(value)?;
    }
    for (node, &demand) in problem.demands.iter().enumerate() {
        if balance[node] != demand {
            return None;
        }
    }
    Some(flow)
}
