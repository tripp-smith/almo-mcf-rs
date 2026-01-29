#[derive(Debug, Clone)]
pub struct CycleCandidate {
    pub ratio: f64,
    pub cycle_edges: Vec<usize>,
}

#[derive(Debug, Default, Clone)]
pub struct MinRatioOracle {
    pub seed: u64,
}
