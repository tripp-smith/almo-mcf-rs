use almo_mcf_core::spanner::oracle::FlowChasingOracle;
use almo_mcf_core::spanner::{DeterministicDynamicSpanner, SpannerBuildParams, SpannerHierarchy};

#[test]
fn spanner_hierarchy_and_oracle_integrate() {
    let tails = vec![0, 1, 2, 0];
    let heads = vec![1, 2, 3, 3];
    let lengths = vec![1.0, 1.0, 1.0, 1.5];
    let mut hierarchy = SpannerHierarchy::build_recursive(SpannerBuildParams {
        node_count: 4,
        tails: &tails,
        heads: &heads,
        lengths: &lengths,
        seed: 11,
        levels: 2,
        rebuild_every: 3,
        instability_budget: 2,
    })
    .unwrap();
    let updates = vec![(0, 1.2, 0.4), (1, 0.9, -0.2)];
    hierarchy.apply_updates(&updates, 0.1, 1.05);
    assert!(hierarchy.should_rebuild());

    let level = &mut hierarchy.levels[0];
    let mut oracle = FlowChasingOracle::new();
    oracle.gradient_scale = 0.5;
    let cycle = oracle
        .find_cycle(&mut level.spanner, 0, 0, 1)
        .expect("cycle found");
    assert!(oracle.ratio(&cycle).is_some());
}

#[test]
fn deterministic_spanner_preserves_stretch_bound() {
    let mut spanner = DeterministicDynamicSpanner::new(4, 2.5, 16, 1);
    let e0 = spanner.insert_edge_with_values(0, 1, 1.0, 0.0);
    let e1 = spanner.insert_edge_with_values(1, 2, 1.0, 0.0);
    let e2 = spanner.insert_edge_with_values(2, 3, 1.0, 0.0);
    let e3 = spanner.insert_edge_with_values(0, 2, 2.0, 0.0);
    let e4 = spanner.insert_edge_with_values(0, 3, 0.5, 0.0);

    for edge_id in [e0, e1, e2, e3, e4] {
        assert!(spanner.embedding_valid(edge_id));
        let ratio = spanner.embedding_ratio(edge_id).unwrap();
        assert!(ratio <= 2.5 + 1e-9);
    }
}

#[test]
fn deterministic_spanner_rebuilds_after_deletion() {
    let mut spanner = DeterministicDynamicSpanner::new(3, 3.0, 8, 1);
    let edge_id = spanner.insert_edge_with_values(0, 1, 1.0, 0.0);
    spanner.insert_edge_with_values(1, 2, 1.0, 0.0);
    assert!(spanner.embedding_valid(edge_id));
    spanner.delete_edge(edge_id);
    assert!(!spanner.embedding_valid(edge_id));
}
