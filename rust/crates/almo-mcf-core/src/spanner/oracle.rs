use crate::spanner::{DynamicSpanner, EmbeddingPath, EmbeddingStep};

#[derive(Debug, Clone, Copy)]
pub struct FlowCycleMetrics {
    pub length: f64,
    pub gradient: f64,
}

#[derive(Debug, Clone)]
pub struct FlowCycle {
    pub steps: Vec<EmbeddingStep>,
    pub metrics: FlowCycleMetrics,
}

#[derive(Debug, Default, Clone)]
pub struct FlowChasingOracle {
    pub gradient_scale: f64,
}

impl FlowChasingOracle {
    pub fn new() -> Self {
        Self {
            gradient_scale: 1.0,
        }
    }

    pub fn find_cycle(
        &self,
        spanner: &mut DynamicSpanner,
        original_edge: usize,
        start: usize,
        end: usize,
    ) -> Option<FlowCycle> {
        let mut steps = spanner.embed_edge_with_bfs(original_edge, start, end)?;
        steps.push(EmbeddingStep::new(original_edge, 1));
        let metrics = self.metrics_for_path(spanner, &steps)?;
        Some(FlowCycle { steps, metrics })
    }

    pub fn metrics_for_path(
        &self,
        spanner: &DynamicSpanner,
        steps: &[EmbeddingStep],
    ) -> Option<FlowCycleMetrics> {
        let mut length = 0.0;
        let mut gradient = 0.0;
        for step in steps {
            let (u, v) = spanner.oriented_endpoints(step.edge, step.dir)?;
            let edge = spanner.edges.get(step.edge)?;
            if !edge.active || u == v {
                return None;
            }
            length += edge.length;
            gradient += (step.dir as f64) * edge.gradient * self.gradient_scale;
        }
        Some(FlowCycleMetrics { length, gradient })
    }

    pub fn ratio(&self, cycle: &FlowCycle) -> Option<f64> {
        if cycle.metrics.length <= 0.0 {
            return None;
        }
        Some(cycle.metrics.gradient / cycle.metrics.length)
    }

    pub fn to_embedding(&self, cycle: &FlowCycle) -> EmbeddingPath {
        EmbeddingPath {
            steps: cycle.steps.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_builds_cycle_and_ratio() {
        let mut spanner = DynamicSpanner::new(3);
        let _e0 = spanner.insert_edge_with_values(0, 1, 1.0, 1.0);
        let _e1 = spanner.insert_edge_with_values(1, 2, 1.0, 2.0);
        let e2 = spanner.insert_edge_with_values(2, 0, 1.0, -1.0);
        let oracle = FlowChasingOracle::new();
        let cycle = oracle.find_cycle(&mut spanner, e2, 2, 0).unwrap();
        assert!(cycle.steps.iter().any(|step| step.edge == e2));
        assert!(oracle.ratio(&cycle).is_some());
        let embedding = oracle.to_embedding(&cycle);
        assert_eq!(embedding.steps.len(), cycle.steps.len());
    }
}
