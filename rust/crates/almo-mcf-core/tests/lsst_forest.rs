use rand::{Rng, SeedableRng};

use almo_mcf_core::graph::{Graph, NodeId};
use almo_mcf_core::trees::forest::DynamicForest;
use almo_mcf_core::trees::hierarchy::HierarchicalTreeChain;
use almo_mcf_core::trees::mwu::{sample_weighted_trees, MwuConfig};
use almo_mcf_core::trees::shift::ShiftableForestCollection;
use almo_mcf_core::trees::{build_random_lsst, LowStretchTree, LsstConfig};

fn random_graph(node_count: usize, edge_prob: f64) -> (Graph, Vec<f64>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut graph = Graph::new(node_count);
    let mut lengths = Vec::new();
    for u in 0..node_count {
        for v in (u + 1)..node_count {
            if rng.gen::<f64>() <= edge_prob {
                let tail = NodeId(u);
                let head = NodeId(v);
                graph.add_edge(tail, head, 0.0, 1.0, 1.0).expect("add edge");
                lengths.push(rng.gen_range(0.5..3.0));
            }
        }
    }
    (graph, lengths)
}

#[test]
fn randomized_lsst_has_bounded_average_stretch() {
    let (graph, lengths) = random_graph(30, 0.25);
    let config = LsstConfig {
        seed: 7,
        stretch_target: 8.0,
        sampling_rounds: 6,
        debug: false,
    };
    let (tree, stretch) = build_random_lsst(&graph, &lengths, config).unwrap();
    let mut total: f64 = 0.0;
    let mut count: f64 = 0.0;
    for value in stretch {
        if value.is_finite() {
            total += value;
            count += 1.0;
        }
    }
    let avg = total / count.max(1.0_f64);
    let log_n = (graph.node_count() as f64).ln().max(1.0);
    let bound = 50.0 * log_n.powi(4);
    assert!(avg.is_finite());
    assert!(avg <= bound);
    assert!(tree
        .average_stretch(&graph_tails(&graph), &graph_heads(&graph), &lengths)
        .is_some());
}

#[test]
fn forest_promote_and_delete_updates_stretch() {
    let tails = vec![0, 1, 2, 3];
    let heads = vec![1, 2, 3, 0];
    let lengths = vec![1.0, 1.0, 1.0, 1.0];
    let tree = LowStretchTree::build_low_stretch(4, &tails, &heads, &lengths, 1).unwrap();
    let mut forest = DynamicForest::new_from_tree(
        4,
        tails.clone(),
        heads.clone(),
        lengths.clone(),
        tree.tree_edges.clone(),
    )
    .unwrap();
    assert!(forest.promote_root(2));
    assert_eq!(forest.tree.root[2], 2);
    let deleted = tree.tree_edges.iter().position(|&x| x).unwrap();
    forest.delete_edge(deleted);
    let stretch = forest.edge_stretch_overestimate(deleted).unwrap();
    assert!(stretch.is_infinite() || stretch >= 1.0);
}

#[test]
fn mwu_sampling_returns_multiple_trees() {
    let tails = vec![0, 0, 1, 2, 2];
    let heads = vec![1, 2, 2, 3, 4];
    let lengths = vec![1.0, 2.0, 1.5, 1.2, 1.1];
    let circulation = vec![1.0, 0.5, 0.0, 0.2, 0.1];
    let trees = sample_weighted_trees(
        5,
        &tails,
        &heads,
        &lengths,
        &circulation,
        3,
        MwuConfig {
            eta: 0.4,
            iterations: 3,
            seed: 9,
        },
    )
    .unwrap();
    assert_eq!(trees.len(), 3);
    for (_, stretch) in trees {
        assert!(stretch.is_finite());
    }
}

#[test]
fn hierarchy_builds_multiple_levels() {
    let tails = vec![0, 1, 2, 3, 0, 2];
    let heads = vec![1, 2, 3, 4, 4, 4];
    let lengths = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.5];
    let chain = HierarchicalTreeChain::build(5, &tails, &heads, &lengths, 3).unwrap();
    assert!(!chain.levels.is_empty());
    assert!(chain.levels.len() <= 3);
}

#[test]
fn shifting_applies_lazy_deletions() {
    let tails = vec![0, 1, 2, 0];
    let heads = vec![1, 2, 3, 3];
    let lengths = vec![1.0, 1.0, 1.0, 1.0];
    let mut collection =
        ShiftableForestCollection::build_deterministic(4, &tails, &heads, &lengths, 2, 2).unwrap();
    collection.mark_deleted(1);
    collection.shift(0).unwrap();
    let forest = collection.current_forest(0).unwrap();
    assert!(forest.is_deleted(1));
}

fn graph_tails(graph: &Graph) -> Vec<u32> {
    graph.edges().map(|(_, edge)| edge.tail.0 as u32).collect()
}

fn graph_heads(graph: &Graph) -> Vec<u32> {
    graph.edges().map(|(_, edge)| edge.head.0 as u32).collect()
}
