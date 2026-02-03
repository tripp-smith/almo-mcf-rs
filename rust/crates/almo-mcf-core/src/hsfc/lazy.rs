use crate::graph::EdgeId;

use super::properties::HSFCWitness;

pub trait LazyOracle {
    fn lazy_gradient(&mut self, edge: EdgeId, base_gradient: f64) -> f64;
    fn lazy_length(&mut self, edge: EdgeId, base_length: f64) -> f64;
}

#[derive(Debug, Clone)]
pub struct LazyHSFCOracle {
    history: Vec<HSFCWitness>,
    max_history: usize,
}

impl LazyHSFCOracle {
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::new(),
            max_history: max_history.max(1),
        }
    }

    pub fn push_witness(&mut self, witness: HSFCWitness) {
        self.history.push(witness);
        if self.history.len() > self.max_history {
            let drain = self.history.len() - self.max_history;
            self.history.drain(0..drain);
        }
    }

    fn correction(&self, edge: EdgeId) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let mut accum = 0.0;
        let mut weight = 1.0;
        for witness in self.history.iter().rev() {
            let value = witness.circulation.get(edge.0).copied().unwrap_or(0.0);
            accum += weight * value;
            weight *= 0.5;
        }
        accum
    }
}

impl LazyOracle for LazyHSFCOracle {
    fn lazy_gradient(&mut self, edge: EdgeId, base_gradient: f64) -> f64 {
        base_gradient + self.correction(edge)
    }

    fn lazy_length(&mut self, edge: EdgeId, base_length: f64) -> f64 {
        base_length.abs() + self.correction(edge).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hsfc::properties::HSFCWitness;

    #[test]
    fn lazy_oracle_applies_recent_corrections() {
        let mut oracle = LazyHSFCOracle::new(2);
        oracle.push_witness(HSFCWitness::new(vec![1.0]));
        oracle.push_witness(HSFCWitness::new(vec![0.5]));
        let value = oracle.lazy_gradient(EdgeId(0), 2.0);
        assert!(value > 2.0);
    }
}
