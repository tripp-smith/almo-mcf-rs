pub fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

pub fn l1_norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v.abs()).sum()
}

pub fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

pub fn scaled_add(target: &mut [f64], scale: f64, values: &[f64]) {
    assert_eq!(target.len(), values.len());
    for (t, v) in target.iter_mut().zip(values.iter()) {
        *t += scale * v;
    }
}
