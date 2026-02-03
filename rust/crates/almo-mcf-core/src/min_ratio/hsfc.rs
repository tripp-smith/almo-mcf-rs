//! Hidden Stable Flow Chasing (HSFC) utilities.
//!
//! Lemma 6.1 (Paper 1): we maintain a sequence of circulations c(t) that witness
//! stability for gradients/lengths without rebuilding. The witness width w(t)
//! (the max |c_e(t)|) may only double on edges that were explicitly updated,
//! allowing lazy updates with amortized rebuilds.

#[derive(Debug, Clone)]
pub struct HSFCState {
    pub circulation: Vec<f64>,
    pub edge_widths: Vec<f64>,
    pub width: f64,
    pub dirty_edges: Vec<bool>,
}

impl HSFCState {
    fn new(edge_count: usize) -> Self {
        Self {
            circulation: vec![0.0; edge_count],
            edge_widths: vec![0.0; edge_count],
            width: 0.0,
            dirty_edges: vec![false; edge_count],
        }
    }
}

#[derive(Debug, Clone)]
pub struct HSFCUpdateResult {
    pub stable: bool,
    pub width: f64,
    pub doubled_on_dirty: bool,
}

#[derive(Debug, Clone)]
pub struct HiddenStableFlowChasing {
    history: Vec<HSFCState>,
    eps: f64,
}

impl HiddenStableFlowChasing {
    pub fn new(edge_count: usize) -> Self {
        let mut history = Vec::new();
        history.push(HSFCState::new(edge_count));
        Self { history, eps: 1e-9 }
    }

    pub fn reset(&mut self, edge_count: usize) {
        self.history.clear();
        self.history.push(HSFCState::new(edge_count));
    }

    pub fn edge_count(&self) -> usize {
        self.history
            .last()
            .map(|state| state.circulation.len())
            .unwrap_or(0)
    }

    pub fn len(&self) -> usize {
        self.history.len()
    }

    pub fn get_circulation(&self, t: usize) -> Option<&Vec<f64>> {
        self.history.get(t).map(|state| &state.circulation)
    }

    pub fn get_width(&self, t: usize) -> Option<f64> {
        self.history.get(t).map(|state| state.width)
    }

    pub fn is_stable(&self, edge: usize, t: usize) -> bool {
        if t == 0 {
            return true;
        }
        let Some(current) = self.history.get(t) else {
            return true;
        };
        let Some(prev) = self.history.get(t.saturating_sub(1)) else {
            return true;
        };
        if edge >= current.edge_widths.len() || edge >= prev.edge_widths.len() {
            return true;
        }
        if current.dirty_edges.get(edge).copied().unwrap_or(false) {
            return true;
        }
        current.edge_widths[edge] <= 2.0 * prev.edge_widths[edge] + self.eps
    }

    pub fn latest(&self) -> &HSFCState {
        self.history
            .last()
            .expect("HSFC should always have an initial state")
    }

    pub fn update(&mut self, dirty_edges: &[usize], witness_values: &[f64]) -> HSFCUpdateResult {
        let prev = self.latest();
        let edge_count = prev.circulation.len();
        let mut circulation = prev.circulation.clone();
        let mut dirty_mask = vec![false; edge_count];
        for &edge_id in dirty_edges {
            if edge_id < edge_count {
                dirty_mask[edge_id] = true;
            }
        }
        for (edge_id, value) in witness_values.iter().enumerate().take(edge_count) {
            circulation[edge_id] = *value;
        }

        let mut edge_widths = Vec::with_capacity(edge_count);
        let mut width = 0.0;
        let mut max_edge = 0;
        for (idx, &value) in circulation.iter().enumerate() {
            let w = value.abs();
            edge_widths.push(w);
            if w > width {
                width = w;
                max_edge = idx;
            }
        }

        let mut stable = true;
        for edge_id in 0..edge_count {
            if dirty_mask[edge_id] {
                continue;
            }
            if edge_widths[edge_id] > 2.0 * prev.edge_widths[edge_id] + self.eps {
                stable = false;
                break;
            }
        }

        let doubled_on_dirty = if width > 2.0 * prev.width + self.eps {
            dirty_mask.get(max_edge).copied().unwrap_or(false)
        } else {
            true
        };
        if !doubled_on_dirty {
            stable = false;
        }

        self.history.push(HSFCState {
            circulation,
            edge_widths,
            width,
            dirty_edges: dirty_mask,
        });

        HSFCUpdateResult {
            stable,
            width,
            doubled_on_dirty,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsfc_initializes_zero_width() {
        let hsfc = HiddenStableFlowChasing::new(3);
        assert_eq!(hsfc.len(), 1);
        assert_eq!(hsfc.get_width(0), Some(0.0));
        assert_eq!(hsfc.get_circulation(0).unwrap(), &vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn hsfc_width_doubles_only_on_dirty_edges() {
        let mut hsfc = HiddenStableFlowChasing::new(3);
        let mut witness = vec![0.0; 3];
        witness[1] = 1.0;
        let update = hsfc.update(&[1], &witness);
        assert!(update.stable);
        assert!(update.doubled_on_dirty);
        assert!(hsfc.is_stable(0, 1));
        assert!(hsfc.is_stable(2, 1));

        let mut witness2 = vec![0.0; 3];
        witness2[0] = 3.0;
        let update2 = hsfc.update(&[0], &witness2);
        assert!(update2.stable);
        assert!(hsfc.is_stable(2, 2));
    }

    #[test]
    fn hsfc_flags_non_dirty_width_jump() {
        let mut hsfc = HiddenStableFlowChasing::new(2);
        let mut witness = vec![0.0; 2];
        witness[0] = 1.0;
        hsfc.update(&[0], &witness);

        let mut witness2 = vec![0.0; 2];
        witness2[1] = 10.0;
        let update2 = hsfc.update(&[], &witness2);
        assert!(!update2.stable);
        assert!(!update2.doubled_on_dirty);
    }
}
