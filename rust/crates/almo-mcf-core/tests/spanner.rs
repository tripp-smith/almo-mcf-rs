use almo_mcf_core::spanner::oracle::FlowChasingOracle;
use almo_mcf_core::spanner::SpannerHierarchy;

#[test]
fn spanner_hierarchy_and_oracle_integrate() {
    let tails = vec![0, 1, 2, 0];
    let heads = vec![1, 2, 3, 3];
    let lengths = vec![1.0, 1.0, 1.0, 1.5];
    let mut hierarchy =
        SpannerHierarchy::build_recursive(4, &tails, &heads, &lengths, 11, 2, 3, 2).unwrap();
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
