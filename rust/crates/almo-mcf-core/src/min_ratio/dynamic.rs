use crate::min_ratio::{
    select_better_candidate, CycleCandidate, MinRatioOracle, OracleQuery, TreeError,
};
use crate::spanner::{DeterministicDynamicSpanner, DynamicSpanner, EmbeddingStep};
use crate::trees::dynamic::DynamicTree;
use crate::trees::{LowStretchTree, TreeBuildMode};

#[derive(Debug, Clone)]
enum EdgeSpanner {
    Randomized(DynamicSpanner),
    Deterministic(DeterministicDynamicSpanner),
}

impl EdgeSpanner {
    fn new(deterministic: bool, node_count: usize) -> Self {
        if deterministic {
            let max_edges = node_count.saturating_mul(8).max(1);
            let rebuild_every = ((node_count as f64).ln().ceil() as usize).max(1);
            EdgeSpanner::Deterministic(DeterministicDynamicSpanner::new(
                node_count,
                4.0,
                max_edges,
                rebuild_every,
            ))
        } else {
            EdgeSpanner::Randomized(DynamicSpanner::new(node_count))
        }
    }

    fn node_count(&self) -> usize {
        match self {
            EdgeSpanner::Randomized(spanner) => spanner.node_count,
            EdgeSpanner::Deterministic(spanner) => spanner.node_count,
        }
    }

    fn rebuild_from_snapshot(
        &mut self,
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        gradients: &[f64],
        lengths: &[f64],
    ) {
        match self {
            EdgeSpanner::Randomized(spanner) => {
                let mut new_spanner = DynamicSpanner::new(node_count);
                for (edge_id, (&tail, &head)) in tails.iter().zip(heads.iter()).enumerate() {
                    let spanner_edge = new_spanner.insert_edge_with_values(
                        tail as usize,
                        head as usize,
                        lengths[edge_id].abs(),
                        gradients[edge_id],
                    );
                    new_spanner.set_embedding(edge_id, vec![EmbeddingStep::new(spanner_edge, 1)]);
                }
                *spanner = new_spanner;
            }
            EdgeSpanner::Deterministic(spanner) => {
                spanner.sync_snapshot(node_count, tails, heads, lengths, gradients);
            }
        }
    }

    fn update_edge_values(&mut self, edge_id: usize, length: f64, gradient: f64) -> bool {
        match self {
            EdgeSpanner::Randomized(spanner) => {
                spanner.update_edge_values(edge_id, length, gradient)
            }
            EdgeSpanner::Deterministic(spanner) => {
                spanner.update_edge_values(edge_id, length, gradient)
            }
        }
    }

    fn embedding_valid(&self, edge_id: usize) -> bool {
        match self {
            EdgeSpanner::Randomized(spanner) => spanner.embedding_valid(edge_id),
            EdgeSpanner::Deterministic(spanner) => spanner.embedding_valid(edge_id),
        }
    }

    fn embedding_steps(&self, edge_id: usize) -> Option<&[EmbeddingStep]> {
        match self {
            EdgeSpanner::Randomized(spanner) => spanner.embedding_steps(edge_id),
            EdgeSpanner::Deterministic(spanner) => spanner.embedding_steps(edge_id),
        }
    }

    fn embed_edge_with_bfs(
        &mut self,
        edge_id: usize,
        start: usize,
        end: usize,
    ) -> Option<Vec<EmbeddingStep>> {
        match self {
            EdgeSpanner::Randomized(spanner) => spanner.embed_edge_with_bfs(edge_id, start, end),
            EdgeSpanner::Deterministic(spanner) => spanner.embed_edge_with_bfs(edge_id, start, end),
        }
    }

    #[cfg(test)]
    fn insert_edge(&mut self, u: usize, v: usize) -> usize {
        match self {
            EdgeSpanner::Randomized(spanner) => spanner.insert_edge(u, v),
            EdgeSpanner::Deterministic(spanner) => spanner.insert_edge_with_values(u, v, 1.0, 0.0),
        }
    }

    #[cfg(test)]
    fn set_embedding(&mut self, edge_id: usize, steps: Vec<EmbeddingStep>) {
        match self {
            EdgeSpanner::Randomized(spanner) => spanner.set_embedding(edge_id, steps),
            EdgeSpanner::Deterministic(_) => {}
        }
    }

    #[cfg(test)]
    fn embedding_endpoints(&self, edge_id: usize) -> Option<(usize, usize)> {
        match self {
            EdgeSpanner::Randomized(spanner) => spanner.embedding_endpoints(edge_id),
            EdgeSpanner::Deterministic(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TreeChainLevel {
    pub oracle: MinRatioOracle,
    pub instability_budget: usize,
    pub max_instability: usize,
    pub last_rebuild: usize,
    pub approx_factor: f64,
    pub needs_rebuild: bool,
}

impl TreeChainLevel {
    pub fn new(
        seed: u64,
        rebuild_every: usize,
        max_instability: usize,
        approx_factor: f64,
        deterministic: bool,
    ) -> Self {
        Self {
            oracle: MinRatioOracle::new_with_mode(seed, rebuild_every, deterministic),
            instability_budget: 0,
            max_instability,
            last_rebuild: 0,
            approx_factor,
            needs_rebuild: false,
        }
    }

    pub fn record_instability(&mut self, amount: usize) {
        self.instability_budget = self.instability_budget.saturating_add(amount);
        if self.instability_budget >= self.max_instability {
            self.instability_budget = 0;
            self.needs_rebuild = true;
        }
    }
}

#[derive(Debug, Clone)]
pub struct TreeChainHierarchy {
    pub seed: u64,
    pub levels: Vec<TreeChainLevel>,
    pub rebuild_every: usize,
}

impl TreeChainHierarchy {
    pub fn new(seed: u64, levels: usize, rebuild_every: usize, max_instability: usize) -> Self {
        Self::new_with_approx(seed, levels, rebuild_every, max_instability, 0.0, false)
    }

    pub fn new_with_approx(
        seed: u64,
        levels: usize,
        rebuild_every: usize,
        max_instability: usize,
        approx_factor: f64,
        deterministic: bool,
    ) -> Self {
        let mut chain = Vec::with_capacity(levels);
        for level in 0..levels {
            chain.push(TreeChainLevel::new(
                seed ^ (0x9e37_79b9_7f4a_7c15u64.wrapping_mul(level as u64 + 1)),
                rebuild_every,
                max_instability,
                approx_factor,
                deterministic,
            ));
        }
        Self {
            seed,
            levels: chain,
            rebuild_every,
        }
    }

    pub fn best_cycle(
        &mut self,
        iter: usize,
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        gradients: &[f64],
        lengths: &[f64],
    ) -> Result<Option<CycleCandidate>, TreeError> {
        let mut best: Option<CycleCandidate> = None;
        let mut best_score: Option<f64> = None;
        let query = OracleQuery {
            iter,
            node_count,
            tails,
            heads,
            gradients,
            lengths,
        };
        for level in &mut self.levels {
            let candidate = level
                .oracle
                .best_cycle_with_rebuild(query, level.needs_rebuild)?;
            if level.needs_rebuild {
                level.needs_rebuild = false;
                level.last_rebuild = iter;
            }
            if let Some(candidate) = candidate {
                let score = candidate.ratio / (1.0 + level.approx_factor);
                if best_score.map(|best| score < best).unwrap_or(true) {
                    best = Some(candidate);
                    best_score = Some(score);
                }
            }
        }
        Ok(best)
    }

    pub fn record_instability(&mut self, amount: usize) {
        for level in &mut self.levels {
            level.record_instability(amount);
        }
    }

    pub fn update_instability_threshold(&mut self, edge_count: usize, exponent: f64) {
        if edge_count == 0 || exponent <= 0.0 {
            return;
        }
        let tau = (edge_count as f64).powf(exponent).ceil().max(1.0) as usize;
        for level in &mut self.levels {
            level.max_instability = level.max_instability.max(tau);
        }
    }
}

#[derive(Debug, Clone)]
pub struct FullDynamicOracle {
    hierarchy: TreeChainHierarchy,
    spanner: EdgeSpanner,
    tree_chain: Option<TreeChain>,
    dynamic_tree: Option<DynamicTree>,
    rebuild_game: RebuildGame,
    amortized: AmortizedTracker,
    last_gradients: Vec<f64>,
    last_lengths: Vec<f64>,
    last_edge_count: usize,
    node_count: usize,
    gradient_change: f64,
    length_factor: f64,
    instability_exponent: f64,
    high_flow_threshold: f64,
    deterministic: bool,
    tree_rebuild_every: usize,
    tree_update_budget: usize,
    tree_length_factor: f64,
}

impl FullDynamicOracle {
    pub fn new(
        seed: u64,
        levels: usize,
        rebuild_every: usize,
        max_instability: usize,
        approx_factor: f64,
        deterministic: bool,
    ) -> Self {
        Self {
            hierarchy: TreeChainHierarchy::new_with_approx(
                seed,
                levels,
                rebuild_every,
                max_instability,
                approx_factor,
                deterministic,
            ),
            spanner: EdgeSpanner::new(deterministic, 0),
            tree_chain: None,
            dynamic_tree: None,
            rebuild_game: RebuildGame::new(max_instability as f64, 1.5),
            amortized: AmortizedTracker::new(),
            last_gradients: Vec::new(),
            last_lengths: Vec::new(),
            last_edge_count: 0,
            node_count: 0,
            gradient_change: 0.5,
            length_factor: 1.25,
            instability_exponent: 0.1,
            high_flow_threshold: 1.0,
            deterministic,
            tree_rebuild_every: rebuild_every.max(1),
            tree_update_budget: max_instability.max(1),
            tree_length_factor: 1.25,
        }
    }

    fn rebuild_spanner(
        &mut self,
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        gradients: &[f64],
        lengths: &[f64],
    ) {
        self.spanner
            .rebuild_from_snapshot(node_count, tails, heads, gradients, lengths);
        self.last_edge_count = tails.len();
        self.node_count = node_count;
    }

    fn rebuild_tree_chain(
        &mut self,
        iter: usize,
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        lengths: &[f64],
    ) -> Result<(), TreeError> {
        let build_mode = if self.deterministic {
            TreeBuildMode::Deterministic
        } else {
            TreeBuildMode::Randomized
        };
        let dynamic_tree = match self.dynamic_tree.take() {
            Some(mut tree) => {
                tree.build_mode = build_mode;
                tree.seed = self.hierarchy.seed;
                tree.update_from_snapshot(iter, node_count, tails, heads, lengths)?;
                tree
            }
            None => DynamicTree::new_with_mode(
                node_count,
                tails.to_vec(),
                heads.to_vec(),
                lengths.to_vec(),
                self.hierarchy.seed,
                self.tree_rebuild_every,
                self.tree_update_budget,
                self.tree_length_factor,
                build_mode,
            )?,
        };
        self.dynamic_tree = Some(dynamic_tree);
        let tree = self
            .dynamic_tree
            .as_ref()
            .expect("dynamic tree should exist")
            .tree
            .clone();
        let chain = TreeChain::new(tree, tails, heads, 3, 8, self.deterministic);
        self.tree_chain = Some(chain);
        Ok(())
    }

    fn ensure_tree_chain(
        &mut self,
        iter: usize,
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        lengths: &[f64],
    ) -> Result<(), TreeError> {
        if self.tree_chain.is_none() || self.dynamic_tree.is_none() {
            return self.rebuild_tree_chain(iter, node_count, tails, heads, lengths);
        }

        let Some(tree_state) = self.dynamic_tree.as_mut() else {
            return self.rebuild_tree_chain(iter, node_count, tails, heads, lengths);
        };
        let rebuilt = tree_state.update_from_snapshot(iter, node_count, tails, heads, lengths)?;
        if rebuilt
            || self
                .tree_chain
                .as_ref()
                .map(|chain| chain.tree.parent.len() != node_count)
                .unwrap_or(true)
        {
            if let Some(chain) = self.tree_chain.as_mut() {
                chain.refresh(tree_state.tree.clone(), tails, heads);
            }
        }
        Ok(())
    }

    fn force_tree_chain_rebuild(
        &mut self,
        iter: usize,
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        lengths: &[f64],
    ) -> Result<(), TreeError> {
        if self.dynamic_tree.is_none() {
            self.rebuild_tree_chain(iter, node_count, tails, heads, lengths)?;
        }
        if let Some(tree_state) = self.dynamic_tree.as_mut() {
            tree_state.force_rebuild(iter)?;
        }
        let tree = self
            .dynamic_tree
            .as_ref()
            .expect("dynamic tree should exist")
            .tree
            .clone();
        if let Some(chain) = self.tree_chain.as_mut() {
            chain.refresh(tree, tails, heads);
        } else {
            let chain = TreeChain::new(tree, tails, heads, 3, 8, self.deterministic);
            self.tree_chain = Some(chain);
        }
        Ok(())
    }

    fn ensure_spanner(
        &mut self,
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        gradients: &[f64],
        lengths: &[f64],
    ) {
        if self.node_count != node_count || self.last_edge_count != tails.len() {
            self.rebuild_spanner(node_count, tails, heads, gradients, lengths);
            return;
        }
        if self.spanner.node_count() != node_count {
            self.rebuild_spanner(node_count, tails, heads, gradients, lengths);
        }
    }

    fn refresh_spanner_values(&mut self, gradients: &[f64], lengths: &[f64]) {
        for (edge_id, (&gradient, &length)) in gradients.iter().zip(lengths.iter()).enumerate() {
            self.spanner
                .update_edge_values(edge_id, length.abs(), gradient);
        }
    }

    pub fn set_instability_exponent(&mut self, exponent: f64) {
        self.instability_exponent = exponent;
    }

    fn record_updates(&mut self, gradients: &[f64], lengths: &[f64]) {
        if self.last_gradients.is_empty() || self.last_lengths.is_empty() {
            self.last_gradients = gradients.to_vec();
            self.last_lengths = lengths.to_vec();
            return;
        }
        let mut instability = 0_usize;
        for ((&prev_g, &prev_l), (&g, &l)) in self
            .last_gradients
            .iter()
            .zip(self.last_lengths.iter())
            .zip(gradients.iter().zip(lengths.iter()))
        {
            if (g - prev_g).abs() > self.gradient_change {
                instability += 1;
                self.amortized.record_update();
            }
            if prev_l > 0.0 && l > 0.0 {
                let ratio = if l > prev_l { l / prev_l } else { prev_l / l };
                if ratio > self.length_factor {
                    instability += 1;
                    self.amortized.record_update();
                }
            }
        }
        if instability > 0 {
            self.hierarchy.record_instability(instability);
        }
        self.last_gradients = gradients.to_vec();
        self.last_lengths = lengths.to_vec();
    }

    fn record_adversary_update(&mut self, instability: usize) -> bool {
        self.rebuild_game.record_update(instability as f64);
        if self.rebuild_game.should_rebuild() {
            self.rebuild_game.on_rebuild();
            return true;
        }
        false
    }

    pub fn identify_high_flow_edges(&self, gradients: &[f64], lengths: &[f64]) -> Vec<usize> {
        gradients
            .iter()
            .zip(lengths.iter())
            .enumerate()
            .filter_map(|(edge_id, (&g, &l))| {
                if l > 0.0 && (g.abs() / l) >= self.high_flow_threshold {
                    Some(edge_id)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn best_cycle(
        &mut self,
        iter: usize,
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        gradients: &[f64],
        lengths: &[f64],
    ) -> Result<Option<CycleCandidate>, TreeError> {
        self.amortized.record_query();
        self.ensure_spanner(node_count, tails, heads, gradients, lengths);
        self.refresh_spanner_values(gradients, lengths);
        self.hierarchy
            .update_instability_threshold(tails.len(), self.instability_exponent);
        self.record_updates(gradients, lengths);
        self.ensure_tree_chain(iter, node_count, tails, heads, lengths)?;
        if self.record_adversary_update(self.hierarchy.levels.len()) {
            self.rebuild_spanner(node_count, tails, heads, gradients, lengths);
            self.force_tree_chain_rebuild(iter, node_count, tails, heads, lengths)?;
        }
        let query = OracleQuery {
            iter,
            node_count,
            tails,
            heads,
            gradients,
            lengths,
        };

        let mut candidates: Vec<(CycleCandidate, f64)> = Vec::new();
        for level in self.hierarchy.levels.iter_mut().rev() {
            let candidate = level
                .oracle
                .best_cycle_with_rebuild(query, level.needs_rebuild)?;
            if level.needs_rebuild {
                level.needs_rebuild = false;
                level.last_rebuild = iter;
            }
            if let Some(candidate) = candidate {
                candidates.push((candidate, level.approx_factor));
            }
        }

        let mut best: Option<CycleCandidate> = None;
        let mut best_score: Option<f64> = None;
        for (candidate, approx_factor) in candidates {
            let expanded = self
                .expand_cycle_edges(&candidate.cycle_edges, tails, heads)
                .unwrap_or_else(|| candidate.cycle_edges.clone());
            let Some((numerator, denominator)) =
                Self::aggregate_cycle_metrics(&expanded, gradients, lengths)
            else {
                continue;
            };
            let ratio = numerator / denominator;
            let score = ratio / (1.0 + approx_factor);
            if best_score.map(|best| score < best).unwrap_or(true) {
                best_score = Some(score);
                best = Some(CycleCandidate {
                    ratio,
                    numerator,
                    denominator,
                    cycle_edges: expanded,
                });
            }
        }

        if let Some(chain) = self.tree_chain.as_ref() {
            if let Some(chain_best) = chain.approx_best_cycle(tails, heads, gradients, lengths) {
                let current_score = best
                    .as_ref()
                    .map(|best| best.ratio)
                    .unwrap_or(f64::INFINITY);
                if chain_best.ratio < current_score {
                    best = Some(chain_best);
                }
            }
        }

        if let Some(candidate) = best.as_ref() {
            let mut valid = true;
            for (edge_id, _) in &candidate.cycle_edges {
                if !self.spanner.embedding_valid(*edge_id) {
                    valid = false;
                    break;
                }
            }
            if !valid {
                self.rebuild_spanner(node_count, tails, heads, gradients, lengths);
            }
        }

        Ok(best)
    }

    pub fn reduce_edge(
        &mut self,
        edge_id: usize,
        tail: usize,
        head: usize,
    ) -> Option<Vec<(usize, i8)>> {
        if let Some(steps) = self.spanner.embedding_steps(edge_id) {
            return Some(steps.iter().map(|step| (step.edge, step.dir)).collect());
        }
        let steps = self.spanner.embed_edge_with_bfs(edge_id, tail, head)?;
        Some(steps.iter().map(|step| (step.edge, step.dir)).collect())
    }

    pub fn edge_embedding(&self, edge_id: usize) -> Option<Vec<(usize, i8)>> {
        self.spanner
            .embedding_steps(edge_id)
            .map(|steps| steps.iter().map(|step| (step.edge, step.dir)).collect())
    }

    fn expand_cycle_edges(
        &mut self,
        cycle_edges: &[(usize, i8)],
        tails: &[u32],
        heads: &[u32],
    ) -> Option<Vec<(usize, i8)>> {
        let mut expanded = Vec::new();
        for &(edge_id, dir) in cycle_edges {
            let embedding =
                self.reduce_edge(edge_id, tails[edge_id] as usize, heads[edge_id] as usize)?;
            for (embedded_edge, embedded_dir) in embedding {
                expanded.push((embedded_edge, dir * embedded_dir));
            }
        }
        Some(expanded)
    }

    fn aggregate_cycle_metrics(
        cycle_edges: &[(usize, i8)],
        gradients: &[f64],
        lengths: &[f64],
    ) -> Option<(f64, f64)> {
        if cycle_edges.is_empty() {
            return None;
        }
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for &(edge_id, dir) in cycle_edges {
            numerator += (dir as f64) * gradients[edge_id];
            denominator += lengths[edge_id].abs();
        }
        if denominator <= 0.0 {
            return None;
        }
        Some((numerator, denominator))
    }
}

#[derive(Debug, Clone)]
pub struct TreeChain {
    pub tree: LowStretchTree,
    pub off_tree_edges: Vec<usize>,
    pub sample_size: usize,
    pub max_off_tree: usize,
    pub deterministic: bool,
}

impl TreeChain {
    pub fn new(
        tree: LowStretchTree,
        tails: &[u32],
        heads: &[u32],
        sample_size: usize,
        max_off_tree: usize,
        deterministic: bool,
    ) -> Self {
        let off_tree_edges =
            collect_off_tree_edges(&tree, tails, heads, max_off_tree, deterministic);
        Self {
            tree,
            off_tree_edges,
            sample_size: sample_size.max(1),
            max_off_tree: max_off_tree.max(1),
            deterministic,
        }
    }

    pub fn refresh(&mut self, tree: LowStretchTree, tails: &[u32], heads: &[u32]) {
        self.tree = tree;
        self.off_tree_edges = collect_off_tree_edges(
            &self.tree,
            tails,
            heads,
            self.max_off_tree,
            self.deterministic,
        );
    }

    pub fn approx_best_cycle(
        &self,
        tails: &[u32],
        heads: &[u32],
        gradients: &[f64],
        lengths: &[f64],
    ) -> Option<CycleCandidate> {
        let mut best: Option<CycleCandidate> = None;
        let mut count = 0;
        for &edge_id in &self.off_tree_edges {
            if count >= self.sample_size {
                break;
            }
            if let Some(candidate) =
                super::score_edge_cycle(&self.tree, edge_id, tails, heads, gradients, lengths)
            {
                best = Some(match best.take() {
                    Some(prev) => select_better_candidate(prev, candidate),
                    None => candidate,
                });
                count += 1;
            }
        }
        best
    }

    pub fn cycle_union_edges(
        &self,
        edges: &[usize],
        tails: &[u32],
        heads: &[u32],
    ) -> Option<Vec<(usize, i8)>> {
        let mut union = Vec::new();
        for &edge_id in edges {
            let cycle = self.tree.fundamental_cycle(edge_id, tails, heads)?;
            union.extend(cycle);
        }
        Some(union)
    }
}

fn collect_off_tree_edges(
    tree: &LowStretchTree,
    tails: &[u32],
    _heads: &[u32],
    max_off_tree: usize,
    deterministic: bool,
) -> Vec<usize> {
    let mut off_tree = Vec::new();
    for edge_id in 0..tails.len() {
        if !tree.tree_edges[edge_id] {
            off_tree.push(edge_id);
        }
        if off_tree.len() >= max_off_tree {
            break;
        }
    }
    if deterministic {
        off_tree.sort_unstable();
    }
    off_tree
}

#[derive(Debug, Clone)]
pub struct CycleRouter {
    pub stretch_bound: f64,
}

#[derive(Debug, Clone)]
pub struct RoutedCycle {
    pub total_length: f64,
    pub edge_count: usize,
    pub within_bound: bool,
}

impl CycleRouter {
    pub fn new(stretch_bound: f64) -> Self {
        Self {
            stretch_bound: stretch_bound.max(1.0),
        }
    }

    pub fn route_cycle(&self, cycle_edges: &[(usize, i8)], lengths: &[f64]) -> Option<RoutedCycle> {
        if cycle_edges.is_empty() {
            return None;
        }
        let mut total_length = 0.0;
        let mut base_length = 0.0;
        for &(edge_id, _) in cycle_edges {
            let length = lengths.get(edge_id)?.abs();
            total_length += length;
            if length > base_length {
                base_length = length;
            }
        }
        let bound = self.stretch_bound * base_length;
        Some(RoutedCycle {
            total_length,
            edge_count: cycle_edges.len(),
            within_bound: total_length <= bound,
        })
    }
}

#[derive(Debug, Clone)]
pub struct RebuildGame {
    budget: f64,
    growth: f64,
    score: f64,
    rebuilds: usize,
}

impl RebuildGame {
    pub fn new(budget: f64, growth: f64) -> Self {
        Self {
            budget: budget.max(1.0),
            growth: growth.max(1.0),
            score: 0.0,
            rebuilds: 0,
        }
    }

    pub fn record_update(&mut self, amount: f64) {
        self.score += amount.max(0.0);
    }

    pub fn should_rebuild(&self) -> bool {
        self.score >= self.budget
    }

    pub fn on_rebuild(&mut self) {
        self.rebuilds += 1;
        self.score = 0.0;
        self.budget *= self.growth;
    }
}

#[derive(Debug, Clone, Default)]
pub struct AmortizedTracker {
    updates: usize,
    queries: usize,
}

impl AmortizedTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_update(&mut self) {
        self.updates += 1;
    }

    pub fn record_query(&mut self) {
        self.queries += 1;
    }

    pub fn amortized_updates_per_query(&self) -> f64 {
        if self.queries == 0 {
            0.0
        } else {
            self.updates as f64 / self.queries as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::min_ratio::{MinRatioOracle, OracleQuery};
    use crate::spanner::{DynamicSpanner, EmbeddingStep};

    #[test]
    fn hierarchy_matches_fallback_cycle() {
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let gradients = vec![1.0, -2.0, 0.5, -3.0];
        let lengths = vec![1.0, 2.0, 1.0, 3.0];
        let mut fallback = MinRatioOracle::new(13, 1);
        let mut hierarchy = TreeChainHierarchy::new(13, 2, 1, 5);
        let fallback_best = fallback
            .best_cycle(OracleQuery {
                iter: 0,
                node_count: 3,
                tails: &tails,
                heads: &heads,
                gradients: &gradients,
                lengths: &lengths,
            })
            .unwrap()
            .unwrap();
        let hierarchy_best = hierarchy
            .best_cycle(0, 3, &tails, &heads, &gradients, &lengths)
            .unwrap()
            .unwrap();
        assert!((fallback_best.ratio - hierarchy_best.ratio).abs() < 1e-9);
        let mut fallback_edges = fallback_best.cycle_edges.clone();
        let mut hierarchy_edges = hierarchy_best.cycle_edges.clone();
        fallback_edges.sort_unstable();
        hierarchy_edges.sort_unstable();
        assert_eq!(fallback_edges, hierarchy_edges);
    }

    #[test]
    fn hierarchy_is_deterministic_under_seed() {
        let tails = vec![0, 1, 2, 0, 1];
        let heads = vec![1, 2, 0, 2, 3];
        let gradients = vec![0.5, -1.5, -2.0, 1.0, 0.25];
        let lengths = vec![1.0, 1.0, 2.0, 2.0, 3.0];
        let mut h1 = TreeChainHierarchy::new(21, 3, 2, 4);
        let mut h2 = TreeChainHierarchy::new(21, 3, 2, 4);
        let best1 = h1
            .best_cycle(0, 4, &tails, &heads, &gradients, &lengths)
            .unwrap()
            .unwrap();
        let best2 = h2
            .best_cycle(0, 4, &tails, &heads, &gradients, &lengths)
            .unwrap()
            .unwrap();
        assert_eq!(best1.cycle_edges, best2.cycle_edges);
        assert!((best1.ratio - best2.ratio).abs() < 1e-12);
    }

    #[test]
    fn instability_budget_triggers_rebuild_tracking() {
        let mut hierarchy = TreeChainHierarchy::new(9, 2, 3, 2);
        hierarchy.record_instability(1);
        assert_eq!(hierarchy.levels[0].last_rebuild, 0);
        hierarchy.record_instability(1);
        assert!(hierarchy.levels[0].needs_rebuild);
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let gradients = vec![1.0, -2.0, 0.5, -3.0];
        let lengths = vec![1.0, 2.0, 1.0, 3.0];
        hierarchy
            .best_cycle(2, 3, &tails, &heads, &gradients, &lengths)
            .unwrap();
        assert_eq!(hierarchy.levels[0].last_rebuild, 2);
    }

    #[test]
    fn instability_budget_resets_after_rebuild() {
        let mut hierarchy = TreeChainHierarchy::new(19, 1, 5, 3);
        hierarchy.record_instability(1);
        assert!(!hierarchy.levels[0].needs_rebuild);
        hierarchy.record_instability(1);
        assert!(!hierarchy.levels[0].needs_rebuild);
        hierarchy.record_instability(1);
        assert!(hierarchy.levels[0].needs_rebuild);

        let tails = vec![0, 1, 2];
        let heads = vec![1, 2, 0];
        let gradients = vec![0.5, -0.3, -0.1];
        let lengths = vec![1.0, 1.0, 1.0];
        hierarchy
            .best_cycle(3, 3, &tails, &heads, &gradients, &lengths)
            .unwrap();
        assert!(!hierarchy.levels[0].needs_rebuild);
        assert_eq!(hierarchy.levels[0].instability_budget, 0);
    }

    #[test]
    fn dynamic_oracle_matches_fallback_on_small_graph() {
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let gradients = vec![1.0, -2.0, 0.5, -3.0];
        let lengths = vec![1.0, 2.0, 1.0, 3.0];
        let mut fallback = MinRatioOracle::new(17, 1);
        let mut dynamic = FullDynamicOracle::new(17, 2, 1, 5, 0.0, false);
        let fallback_best = fallback
            .best_cycle(OracleQuery {
                iter: 0,
                node_count: 3,
                tails: &tails,
                heads: &heads,
                gradients: &gradients,
                lengths: &lengths,
            })
            .unwrap()
            .unwrap();
        let dynamic_best = dynamic
            .best_cycle(0, 3, &tails, &heads, &gradients, &lengths)
            .unwrap()
            .unwrap();
        assert!((fallback_best.ratio - dynamic_best.ratio).abs() < 1e-9);
        let mut fallback_edges = fallback_best.cycle_edges.clone();
        let mut dynamic_edges = dynamic_best.cycle_edges.clone();
        fallback_edges.sort_unstable();
        dynamic_edges.sort_unstable();
        assert_eq!(fallback_edges, dynamic_edges);
    }

    #[test]
    fn dynamic_oracle_tracks_updates_and_stays_stable() {
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let gradients = vec![1.0, -2.0, 0.5, -3.0];
        let lengths = vec![1.0, 2.0, 1.0, 3.0];
        let mut dynamic = FullDynamicOracle::new(31, 3, 2, 2, 0.1, false);
        let first = dynamic
            .best_cycle(0, 3, &tails, &heads, &gradients, &lengths)
            .unwrap()
            .unwrap();
        let updated_gradients = vec![1.6, -2.4, 0.4, -2.8];
        let updated_lengths = vec![1.4, 2.8, 1.1, 2.9];
        let second = dynamic
            .best_cycle(1, 3, &tails, &heads, &updated_gradients, &updated_lengths)
            .unwrap()
            .unwrap();
        assert!(!first.cycle_edges.is_empty());
        assert!(!second.cycle_edges.is_empty());
    }

    #[test]
    fn dynamic_oracle_rebuilds_invalid_embeddings() {
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let gradients = vec![1.0, -2.0, 0.5, -3.0];
        let lengths = vec![1.0, 2.0, 1.0, 3.0];
        let mut dynamic = FullDynamicOracle::new(41, 2, 1, 5, 0.0, false);
        dynamic
            .best_cycle(0, 3, &tails, &heads, &gradients, &lengths)
            .unwrap();
        let mut spanner = DynamicSpanner::new(3);
        let edge = spanner.insert_edge(0, 1);
        spanner.set_embedding(0, vec![crate::spanner::EmbeddingStep::new(edge, 1)]);
        dynamic.spanner = EdgeSpanner::Randomized(spanner);
        assert!(!dynamic.spanner.embedding_valid(2));
        dynamic
            .best_cycle(1, 3, &tails, &heads, &gradients, &lengths)
            .unwrap();
        assert!(dynamic.spanner.embedding_valid(2));
    }

    #[test]
    fn dynamic_oracle_is_deterministic_across_seeds() {
        let tails = vec![0, 1, 2, 0, 1];
        let heads = vec![1, 2, 0, 2, 3];
        let gradients = vec![0.5, -1.5, -2.0, 1.0, 0.25];
        let lengths = vec![1.0, 1.0, 2.0, 2.0, 3.0];
        let mut oracle_a = FullDynamicOracle::new(7, 2, 2, 4, 0.0, true);
        let mut oracle_b = FullDynamicOracle::new(99, 2, 2, 4, 0.0, true);
        let best_a = oracle_a
            .best_cycle(0, 4, &tails, &heads, &gradients, &lengths)
            .unwrap()
            .unwrap();
        let best_b = oracle_b
            .best_cycle(0, 4, &tails, &heads, &gradients, &lengths)
            .unwrap()
            .unwrap();
        assert_eq!(best_a.cycle_edges, best_b.cycle_edges);
        assert!((best_a.ratio - best_b.ratio).abs() < 1e-12);
    }

    #[test]
    fn tree_chain_refreshes_after_length_shift() {
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let gradients = vec![1.0, -2.0, 0.5, -3.0];
        let lengths = vec![1.0, 2.0, 1.0, 3.0];
        let mut dynamic = FullDynamicOracle::new(71, 2, 1, 2, 0.0, true);
        dynamic
            .best_cycle(0, 3, &tails, &heads, &gradients, &lengths)
            .unwrap();
        let updated_lengths = vec![1.5, 4.5, 1.1, 3.4];
        let cycle = dynamic
            .best_cycle(1, 3, &tails, &heads, &gradients, &updated_lengths)
            .unwrap()
            .unwrap();
        assert!(!cycle.cycle_edges.is_empty());
    }

    #[test]
    fn dynamic_oracle_exposes_edge_embeddings() {
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let gradients = vec![1.0, -2.0, 0.5, -3.0];
        let lengths = vec![1.0, 2.0, 1.0, 3.0];
        let mut dynamic = FullDynamicOracle::new(51, 2, 1, 5, 0.0, false);
        dynamic
            .best_cycle(0, 3, &tails, &heads, &gradients, &lengths)
            .unwrap();
        let embedding = dynamic.edge_embedding(1).expect("embedding should exist");
        assert_eq!(embedding, vec![(1, 1)]);
    }

    #[test]
    fn dynamic_oracle_reduces_edges_with_embeddings() {
        let tails = vec![0, 1, 2];
        let heads = vec![1, 2, 0];
        let gradients = vec![0.2, -0.1, 0.3];
        let lengths = vec![1.0, 1.0, 1.0];
        let mut dynamic = FullDynamicOracle::new(59, 2, 1, 5, 0.0, false);
        dynamic
            .best_cycle(0, 3, &tails, &heads, &gradients, &lengths)
            .unwrap();
        let embedding = dynamic
            .reduce_edge(1, tails[1] as usize, heads[1] as usize)
            .expect("embedding should exist");
        assert_eq!(embedding, vec![(1, 1)]);
    }

    #[test]
    fn dynamic_oracle_builds_reduction_for_missing_embedding() {
        let mut dynamic = FullDynamicOracle::new(61, 1, 1, 2, 0.0, false);
        dynamic.spanner = EdgeSpanner::Randomized(DynamicSpanner::new(3));
        let e0 = dynamic.spanner.insert_edge(0, 1);
        let e1 = dynamic.spanner.insert_edge(1, 2);
        let embedding = dynamic
            .reduce_edge(7, 0, 2)
            .expect("bfs should find a path");
        assert_eq!(embedding, vec![(e0, 1), (e1, 1)]);
        assert!(dynamic.spanner.embedding_valid(7));
        assert_eq!(dynamic.spanner.embedding_endpoints(7), Some((0, 2)));
    }

    #[test]
    fn instability_threshold_uses_edge_exponent() {
        let mut hierarchy = TreeChainHierarchy::new(7, 2, 1, 1);
        hierarchy.update_instability_threshold(1024, 0.1);
        let threshold = hierarchy.levels[0].max_instability;
        assert!(threshold >= 2);
        hierarchy.update_instability_threshold(4, 0.1);
        assert_eq!(hierarchy.levels[0].max_instability, threshold);
    }

    #[test]
    fn dynamic_oracle_expands_cycle_edges_with_embeddings() {
        let tails = vec![0, 1, 0];
        let heads = vec![1, 2, 2];
        let gradients = vec![0.5, -1.5, 0.2];
        let lengths = vec![1.0, 2.0, 1.0];
        let mut dynamic = FullDynamicOracle::new(73, 1, 1, 3, 0.0, false);
        dynamic
            .best_cycle(0, 3, &tails, &heads, &gradients, &lengths)
            .unwrap();
        dynamic
            .spanner
            .set_embedding(0, vec![EmbeddingStep::new(1, 1), EmbeddingStep::new(2, 1)]);
        let expanded = dynamic
            .expand_cycle_edges(&[(0, 1)], &tails, &heads)
            .expect("expansion should succeed");
        assert_eq!(expanded, vec![(1, 1), (2, 1)]);
    }

    #[test]
    fn dynamic_oracle_aggregates_cycle_metrics() {
        let gradients = vec![1.5, -2.0, 0.5];
        let lengths = vec![2.0, 3.0, 1.0];
        let metrics = FullDynamicOracle::aggregate_cycle_metrics(
            &[(0, 1), (1, -1), (2, 1)],
            &gradients,
            &lengths,
        )
        .expect("metrics should exist");
        assert!((metrics.0 - 4.0).abs() < 1e-9);
        assert!((metrics.1 - 6.0).abs() < 1e-9);
    }

    #[test]
    fn tree_chain_returns_best_cycle() {
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let gradients = vec![1.0, -2.0, 0.5, -3.0];
        let lengths = vec![1.0, 2.0, 1.0, 3.0];
        let tree = LowStretchTree::build_low_stretch(3, &tails, &heads, &lengths, 2).unwrap();
        let chain = TreeChain::new(tree, &tails, &heads, 2, 4, false);
        let candidate = chain
            .approx_best_cycle(&tails, &heads, &gradients, &lengths)
            .expect("candidate should exist");
        assert!(!candidate.cycle_edges.is_empty());
    }

    #[test]
    fn tree_chain_cycle_union_collects_edges() {
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let lengths = vec![1.0, 2.0, 1.0, 3.0];
        let tree = LowStretchTree::build_low_stretch(3, &tails, &heads, &lengths, 6).unwrap();
        let chain = TreeChain::new(tree, &tails, &heads, 2, 4, false);
        let edges = chain.off_tree_edges.clone();
        let union = chain
            .cycle_union_edges(&edges, &tails, &heads)
            .expect("union should exist");
        assert!(!union.is_empty());
    }

    #[test]
    fn cycle_router_checks_length_bounds() {
        let lengths = vec![1.0, 2.0, 3.0];
        let router = CycleRouter::new(2.0);
        let routed = router
            .route_cycle(&[(0, 1), (1, -1), (2, 1)], &lengths)
            .expect("routed should exist");
        assert!(routed.within_bound);
        assert_eq!(routed.edge_count, 3);
    }

    #[test]
    fn rebuild_game_triggers_and_resets() {
        let mut game = RebuildGame::new(3.0, 2.0);
        game.record_update(1.0);
        assert!(!game.should_rebuild());
        game.record_update(2.0);
        assert!(game.should_rebuild());
        game.on_rebuild();
        assert!(!game.should_rebuild());
    }

    #[test]
    fn amortized_tracker_counts_updates_and_queries() {
        let mut tracker = AmortizedTracker::new();
        tracker.record_update();
        tracker.record_update();
        tracker.record_query();
        tracker.record_query();
        assert!((tracker.amortized_updates_per_query() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn high_flow_edges_respect_threshold() {
        let mut oracle = FullDynamicOracle::new(3, 1, 1, 1, 0.0, false);
        oracle.high_flow_threshold = 0.5;
        let gradients = vec![1.0, 0.1, 2.0];
        let lengths = vec![1.0, 1.0, 10.0];
        let high_flow = oracle.identify_high_flow_edges(&gradients, &lengths);
        assert_eq!(high_flow, vec![0]);
    }
}
