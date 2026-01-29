#[derive(Debug, Default, Clone)]
pub struct IpmState {
    pub flow: Vec<f64>,
    pub potential: f64,
}

#[derive(Debug, Default, Clone)]
pub struct IpmStats {
    pub iterations: usize,
    pub last_step_size: f64,
}
