use crate::graph::EdgeId;
use std::collections::HashSet;

#[derive(Debug, Clone, Default)]
pub struct UpdateBatch {
    pub dirty_edges: HashSet<EdgeId>,
}

#[derive(Debug, Clone)]
pub struct DeterministicMode {
    shifts: Vec<usize>,
    shift_count: usize,
}

impl DeterministicMode {
    pub fn new(levels: usize, shift_count: usize) -> Self {
        Self {
            shifts: vec![0; levels],
            shift_count: shift_count.max(1),
        }
    }

    pub fn shift_and_update(&mut self, level: usize, updates: &UpdateBatch) -> HashSet<EdgeId> {
        if let Some(shift) = self.shifts.get_mut(level) {
            *shift = (*shift + 1) % self.shift_count;
        }
        updates.dirty_edges.clone()
    }

    pub fn active_shift(&self, level: usize) -> Option<usize> {
        self.shifts.get(level).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_shifts_cycle() {
        let mut mode = DeterministicMode::new(1, 2);
        let batch = UpdateBatch::default();
        mode.shift_and_update(0, &batch);
        assert_eq!(mode.active_shift(0), Some(1));
        mode.shift_and_update(0, &batch);
        assert_eq!(mode.active_shift(0), Some(0));
    }
}
