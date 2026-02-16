#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Win,
    Loss,
}

#[derive(Debug, Clone)]
pub struct RebuildingGame {
    levels: usize,
    wins: Vec<usize>,
    losses: Vec<usize>,
}

impl RebuildingGame {
    pub fn new(levels: usize) -> Self {
        Self {
            levels: levels.max(1),
            wins: vec![0; levels.max(1)],
            losses: vec![0; levels.max(1)],
        }
    }

    pub fn record(&mut self, level: usize, outcome: Outcome) {
        let idx = level.min(self.levels - 1);
        match outcome {
            Outcome::Win => self.wins[idx] = self.wins[idx].saturating_add(1),
            Outcome::Loss => self.losses[idx] = self.losses[idx].saturating_add(1),
        }
    }

    pub fn force_rebuild_level(&self) -> Option<usize> {
        (0..self.levels)
            .rev()
            .find(|&level| self.losses[level] > self.wins[level].saturating_add(1))
    }
}
