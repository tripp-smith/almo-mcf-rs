use crate::graph::EdgeId;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Loss {
    pub level: usize,
}

#[derive(Debug, Clone)]
pub struct LevelState {
    pub round_count: u32,
    pub fix_count: u32,
    pub max_rounds: u32,
    pub max_fixes: u32,
}

impl LevelState {
    pub fn new(max_rounds: u32, max_fixes: u32) -> Self {
        Self {
            round_count: 0,
            fix_count: 0,
            max_rounds: max_rounds.max(1),
            max_fixes: max_fixes.max(1),
        }
    }

    pub fn record_round(&mut self) -> bool {
        self.round_count = self.round_count.saturating_add(1);
        self.round_count >= self.max_rounds
    }

    pub fn record_fix(&mut self) -> bool {
        self.fix_count = self.fix_count.saturating_add(1);
        self.fix_count >= self.max_fixes
    }

    pub fn reset(&mut self) {
        self.round_count = 0;
        self.fix_count = 0;
    }
}

#[derive(Debug, Clone)]
pub struct RebuildingGame {
    pub levels: Vec<LevelState>,
}

impl RebuildingGame {
    pub fn new(levels: usize, max_rounds: u32, max_fixes: u32) -> Self {
        let levels = (0..levels)
            .map(|_| LevelState::new(max_rounds, max_fixes))
            .collect();
        Self { levels }
    }

    pub fn advance_round(&mut self, level: usize) -> bool {
        self.levels
            .get_mut(level)
            .map(|state| state.record_round())
            .unwrap_or(false)
    }

    pub fn handle_loss(&mut self, level: usize) -> Vec<usize> {
        let mut rebuilds = Vec::new();
        for idx in level..self.levels.len() {
            let needs_rebuild = self
                .levels
                .get_mut(idx)
                .map(|state| state.record_fix())
                .unwrap_or(false);
            if needs_rebuild {
                rebuilds.push(idx);
                if let Some(state) = self.levels.get_mut(idx) {
                    state.reset();
                }
            } else {
                break;
            }
        }
        rebuilds
    }

    pub fn mark_rebuild(&mut self, level: usize) {
        if let Some(state) = self.levels.get_mut(level) {
            state.reset();
        }
    }
}

#[derive(Debug, Clone)]
pub struct LossContext {
    pub level: usize,
    pub dirty_edges: HashSet<EdgeId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebuild_game_propagates_losses() {
        let mut game = RebuildingGame::new(2, 3, 2);
        assert!(game.handle_loss(0).is_empty());
        let rebuilds = game.handle_loss(0);
        assert_eq!(rebuilds, vec![0]);
    }
}
