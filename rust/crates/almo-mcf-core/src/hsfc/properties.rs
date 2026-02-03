use crate::graph::EdgeId;
use std::collections::HashSet;

const EPS: f64 = 1e-10;

#[derive(Debug, Clone)]
pub struct HSFCWitness {
    pub circulation: Vec<f64>,
    pub widths: Vec<f64>,
    pub global_width: f64,
}

impl HSFCWitness {
    pub fn new(circulation: Vec<f64>) -> Self {
        let mut widths = Vec::with_capacity(circulation.len());
        let mut global_width = 0.0;
        for &value in &circulation {
            let width = value.abs();
            widths.push(width);
            if width > global_width {
                global_width = width;
            }
        }
        Self {
            circulation,
            widths,
            global_width,
        }
    }

    pub fn check_stability(&self, prev: &HSFCWitness, dirty_edges: &HashSet<EdgeId>) -> bool {
        if self.circulation.len() != prev.circulation.len() {
            return false;
        }

        let mut max_edge = 0_usize;
        let mut max_width = 0.0;
        for (idx, &width) in self.widths.iter().enumerate() {
            if width > max_width {
                max_width = width;
                max_edge = idx;
            }
        }

        for (edge_id, (&width, &prev_width)) in
            self.widths.iter().zip(prev.widths.iter()).enumerate()
        {
            if dirty_edges.contains(&EdgeId(edge_id)) {
                continue;
            }
            if width > 2.0 * prev_width + EPS {
                return false;
            }
        }

        if self.global_width > 2.0 * prev.global_width + EPS {
            return dirty_edges.contains(&EdgeId(max_edge));
        }
        true
    }
}

#[derive(Debug, Clone)]
pub struct HSFCSequence {
    witnesses: Vec<HSFCWitness>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HSFCStability {
    Stable,
    Unstable,
}

impl HSFCSequence {
    pub fn new(edge_count: usize) -> Self {
        let initial = HSFCWitness::new(vec![0.0; edge_count]);
        Self {
            witnesses: vec![initial],
        }
    }

    pub fn len(&self) -> usize {
        self.witnesses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.witnesses.is_empty()
    }

    pub fn latest(&self) -> &HSFCWitness {
        self.witnesses
            .last()
            .expect("HSFCSequence should have an initial witness")
    }

    pub fn witness(&self, index: usize) -> Option<&HSFCWitness> {
        self.witnesses.get(index)
    }

    pub fn append(&mut self, witness: HSFCWitness, dirty_edges: &HashSet<EdgeId>) -> HSFCStability {
        let stable = witness.check_stability(self.latest(), dirty_edges);
        self.witnesses.push(witness);
        if stable {
            HSFCStability::Stable
        } else {
            HSFCStability::Unstable
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn witness_stability_respects_dirty_edges() {
        let mut prev = HSFCWitness::new(vec![0.5, 0.5]);
        prev.global_width = 0.5;
        let next = HSFCWitness::new(vec![1.5, 0.5]);
        let mut dirty = HashSet::new();
        dirty.insert(EdgeId(0));
        assert!(next.check_stability(&prev, &dirty));
        let clean = HashSet::new();
        assert!(!next.check_stability(&prev, &clean));
    }

    #[test]
    fn sequence_appends_and_tracks_length() {
        let mut seq = HSFCSequence::new(2);
        let mut dirty = HashSet::new();
        dirty.insert(EdgeId(1));
        let witness = HSFCWitness::new(vec![0.0, 2.0]);
        let status = seq.append(witness, &dirty);
        assert_eq!(status, HSFCStability::Stable);
        assert_eq!(seq.len(), 2);
    }
}
