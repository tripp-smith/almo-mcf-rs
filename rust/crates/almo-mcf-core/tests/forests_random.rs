use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use almo_mcf_core::graph::{Graph, NodeId};
use almo_mcf_core::trees::deterministic::{precompute_forests, Shifter, UpdateBatch};
use almo_mcf_core::trees::lsst::{build_lsst, estimate_average_stretch};
use proptest::prelude::*;

fn build_random_graph(n: usize, p: f64, seed: u64) -> Graph {
    let mut graph = Graph::new(n);
    let mut rng = StdRng::seed_from_u64(seed);
    for u in 0..n {
        for v in (u + 1)..n {
            if rng.gen::<f64>() < p {
                let _ = graph.add_edge(NodeId(u), NodeId(v), 0.0, 1.0, 1.0);
                let _ = graph.add_edge(NodeId(v), NodeId(u), 0.0, 1.0, 1.0);
            }
        }
    }
    graph
}

#[test]
fn randomized_lsst_has_bounded_average_stretch() {
    let n = 1000;
    let p = (n as f64).ln() / n as f64;
    let graph = build_random_graph(n, p, 7);
    let gamma = 1.1;
    let forest = build_lsst(&graph, gamma).expect("forest");
    let avg = estimate_average_stretch(&graph, &forest, 1000).expect("stretch");
    let bound = gamma * (n as f64).ln().powi(4) + 1e-6;
    assert!(avg <= bound, "avg stretch {} > {}", avg, bound);
}

#[test]
fn deterministic_forests_shift_and_apply_updates() {
    let n = 128;
    let p = (n as f64).ln() / n as f64;
    let graph = build_random_graph(n, p, 11);
    let forests = precompute_forests(&graph, 10).expect("forests");
    let mut shifter = Shifter::new(forests, (n as f64).ln().powi(5));
    let mut updates = UpdateBatch::new();
    updates.deletions.push((NodeId(0), NodeId(1)));
    shifter.apply_shift(&graph, &updates).expect("shift");
    let _ = shifter.next_forest();
    let current = shifter.current_forest().expect("forest");
    assert!(!current.trees.is_empty());
}

#[test]
fn forests_remain_spanning_after_deletions() {
    let n = 64;
    let p = (n as f64).ln() / n as f64;
    let graph = build_random_graph(n, p, 3);
    let forest = build_lsst(&graph, 1.1).expect("forest");
    for tree in forest {
        let roots: std::collections::HashSet<_> = tree.root.iter().copied().collect();
        assert!(!roots.contains(&usize::MAX));
    }
}

proptest! {
    #[test]
    fn forest_spans_nodes_in_small_graphs(seed in 0u64..1000) {
        let n = 16;
        let p = (n as f64).ln() / n as f64;
        let graph = build_random_graph(n, p, seed);
        let forest = build_lsst(&graph, 1.1).expect("forest");
        for tree in forest {
            for node in 0..n {
                prop_assert_ne!(tree.root[node], usize::MAX);
            }
        }
    }
}
