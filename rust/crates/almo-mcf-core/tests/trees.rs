use almo_mcf_core::trees::dynamic::DynamicTree;
use almo_mcf_core::trees::LowStretchTree;

#[test]
fn dynamic_tree_rebuilds_on_schedule() {
    let tails = vec![0, 1, 2, 3];
    let heads = vec![1, 2, 3, 0];
    let lengths = vec![1.0, 1.2, 0.8, 1.5];
    let mut tree = DynamicTree::new(4, tails, heads, lengths, 7, 3, 2).unwrap();
    tree.update_length(0, 1.3);
    tree.update_length(1, 1.1);
    assert!(tree.should_rebuild(2));
    tree.rebuild(9).unwrap();
    assert!(tree.tree.path_length(0, 2).is_some());
}

#[test]
fn fundamental_cycle_returns_off_tree_edge() {
    let tails = vec![0, 1, 2, 0];
    let heads = vec![1, 2, 0, 2];
    let lengths = vec![1.0, 1.0, 1.0, 0.7];
    let tree = LowStretchTree::build_low_stretch(3, &tails, &heads, &lengths, 2).unwrap();
    let cycle = tree.fundamental_cycle(3, &tails, &heads).unwrap();
    assert!(cycle.iter().any(|(edge_id, _)| *edge_id == 3));
}
