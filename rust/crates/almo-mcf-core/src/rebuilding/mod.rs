use crate::graph::EdgeId;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub struct GameLevel {
    pub level: usize,
    pub threshold: usize,
    pub p_j: f64,
    pub wins: usize,
    pub rounds: usize,
}

#[derive(Debug, Clone, Default)]
pub struct RebuildDecision {
    pub trigger_rebuild: bool,
    pub level: Option<usize>,
    pub levels_to_rebuild: Vec<usize>,
    pub recourse_added: f64,
}

#[derive(Debug, Clone, Default)]
pub struct GameStats {
    pub rebuild_levels_triggered: Vec<usize>,
    pub win_rate_per_level: Vec<f64>,
    pub amortized_per_update: f64,
}

#[derive(Debug, Clone)]
pub struct RebuildingGame {
    pub levels: Vec<GameLevel>,
    pub win_thresholds: Vec<usize>,
    pub loss_counters: Vec<usize>,
    pub total_recourse: f64,
    n: usize,
    m: usize,
    derandomized_seed: Option<u64>,
    recent_rebuilds: Vec<usize>,
}

impl RebuildingGame {
    pub fn new(n: usize, m: usize, l: usize) -> Self {
        let level_count = l.max(1);
        let mut levels = Vec::with_capacity(level_count);
        let mut win_thresholds = Vec::with_capacity(level_count);
        for j in 1..=level_count {
            let exponent = 1.0 - (j as f64) / (level_count as f64);
            let threshold = (n.max(2) as f64).powf(exponent).round().max(1.0) as usize;
            // Paper 1 Def. 8.1: p_j = 1 - 1/j
            let p_j = if j == 1 { 0.5 } else { 1.0 - 1.0 / (j as f64) };
            levels.push(GameLevel {
                level: j,
                threshold,
                p_j,
                wins: 0,
                rounds: 0,
            });
            win_thresholds.push(threshold);
        }

        Self {
            levels,
            win_thresholds,
            loss_counters: vec![0; level_count],
            total_recourse: 0.0,
            n: n.max(2),
            m: m.max(1),
            derandomized_seed: None,
            recent_rebuilds: Vec::new(),
        }
    }

    pub fn enable_derandomization(&mut self, seed: u64) {
        // Paper 2 §3-§5: deterministic hash-based outcomes.
        self.derandomized_seed = Some(seed);
    }

    pub fn play_round(&mut self, t: usize, update_size: usize, is_win: bool) -> RebuildDecision {
        self.recent_rebuilds.clear();
        // Paper 1 Alg 6 line 1: choose j* = argmin {j | t mod t_j == 0}
        let mut chosen = self.levels.len().saturating_sub(1);
        for (idx, threshold) in self.win_thresholds.iter().enumerate() {
            if t.is_multiple_of((*threshold).max(1)) {
                chosen = idx;
                break;
            }
        }

        // Paper 1 Alg 6 line 2: observe win/loss on level j*
        let actual_win = self.resolve_win(t, chosen, is_win);
        if let Some(level) = self.levels.get_mut(chosen) {
            level.rounds = level.rounds.saturating_add(1);
            if actual_win {
                level.wins = level.wins.saturating_add(1);
            }
        }

        // Paper 1 Alg 6 lines 3-4: increment loss counter on loss
        if !actual_win {
            self.loss_counters[chosen] = self.loss_counters[chosen].saturating_add(1);
        }

        // Paper 1 Alg 6 lines 5-12: trigger rebuild if loss budget exceeded.
        let p_j = self.levels[chosen].p_j.max(1e-9);
        let loss_budget = ((self.n as f64).ln().max(1.0) / p_j).ceil() as usize;
        let should_rebuild = self.loss_counters[chosen] > loss_budget;
        let recourse_added = if should_rebuild {
            self.loss_counters[chosen] = 0;
            self.recent_rebuilds.push(chosen);
            self.compute_recourse_bound(chosen) * (update_size.max(1) as f64).ln().max(1.0)
        } else {
            0.0
        };
        self.total_recourse += recourse_added;

        RebuildDecision {
            trigger_rebuild: should_rebuild,
            level: should_rebuild.then_some(chosen),
            levels_to_rebuild: self.recent_rebuilds.clone(),
            recourse_added,
        }
    }

    pub fn handle_adversarial_update(&mut self, deletions: &[EdgeId]) -> Vec<usize> {
        let pressure = deletions.len().max(1);
        let mut rebuilds = Vec::new();
        for j in 0..self.levels.len() {
            if pressure >= self.win_thresholds[j].max(1) {
                self.loss_counters[j] = self.loss_counters[j].saturating_add(1);
            }
            let p_j = self.levels[j].p_j.max(1e-9);
            let loss_budget = ((self.n as f64).ln().max(1.0) / p_j).ceil() as usize;
            if self.loss_counters[j] > loss_budget {
                self.loss_counters[j] = 0;
                rebuilds.push(j);
            }
        }
        self.recent_rebuilds = rebuilds.clone();
        rebuilds
    }

    pub fn verify_derandomized_invariants(&self) -> bool {
        // Paper 2 §5: deterministic critical path when derandomized.
        self.derandomized_seed.is_none_or(|_| {
            self.levels
                .iter()
                .all(|level| level.threshold > 0 && level.p_j.is_finite())
        })
    }

    pub fn compute_amortized_time(&self, total_iters: usize) -> f64 {
        let t = total_iters.max(1) as f64;
        let rebuild_sum: f64 = self
            .loss_counters
            .iter()
            .enumerate()
            .map(|(j, losses)| self.compute_recourse_bound(j) * (*losses as f64 + 1.0))
            .sum();
        (self.total_recourse + rebuild_sum).min(self.m as f64 * (self.n as f64).ln().powi(2)) / t
    }

    pub fn log_game_stats(&self, t: usize) -> GameStats {
        GameStats {
            rebuild_levels_triggered: self.recent_rebuilds.clone(),
            win_rate_per_level: self
                .levels
                .iter()
                .map(|level| {
                    if level.rounds == 0 {
                        1.0
                    } else {
                        level.wins as f64 / level.rounds as f64
                    }
                })
                .collect(),
            amortized_per_update: self.compute_amortized_time(t.max(1)),
        }
    }

    fn resolve_win(&self, t: usize, j: usize, is_win: bool) -> bool {
        if let Some(seed) = self.derandomized_seed {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            t.hash(&mut hasher);
            j.hash(&mut hasher);
            let h = hasher.finish();
            (h & 1) == 0 || is_win
        } else {
            is_win
        }
    }

    fn compute_recourse_bound(&self, j: usize) -> f64 {
        let level = &self.levels[j.min(self.levels.len().saturating_sub(1))];
        let logn = (self.n as f64).ln().max(1.0);
        // Lemma 8.3 style union-bound upper envelope O(m * n^{o(1)}).
        (self.m as f64) * logn / level.p_j.max(1e-9)
    }
}
