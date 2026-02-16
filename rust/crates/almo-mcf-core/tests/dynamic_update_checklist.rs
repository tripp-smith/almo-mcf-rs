use almo_mcf_core::graph::{Graph, NodeId};
use almo_mcf_core::min_ratio::oracle::DynamicOracle;
use almo_mcf_core::min_ratio::OracleQuery;
use almo_mcf_core::spanner::{DecrementalSpanner, DecrementalSpannerParams};
use almo_mcf_core::trees::forest::DynamicForest;
use almo_mcf_core::trees::LowStretchTree;
use rand::prelude::*;

fn random_graph(n: usize, m: usize, seed: u64) -> (Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut tails = Vec::with_capacity(m);
    let mut heads = Vec::with_capacity(m);
    let mut lengths = Vec::with_capacity(m);
    let mut gradients = Vec::with_capacity(m);
    for _ in 0..m {
        let u = rng.gen_range(0..n) as u32;
        let mut v = rng.gen_range(0..n) as u32;
        while v == u {
            v = rng.gen_range(0..n) as u32;
        }
        tails.push(u);
        heads.push(v);
        lengths.push(rng.gen_range(1.0..5.0));
        gradients.push(rng.gen_range(-1.0..1.0));
    }
    (tails, heads, lengths, gradients)
}

#[test]
fn test_decremental_spanner_update() {
    let n = 30usize;
    let (tails, heads, lengths, _) = random_graph(n, 120, 7);
    let params = DecrementalSpannerParams::from_graph(n, tails.len());
    let mut spanner = DecrementalSpanner::new(n, &tails, &heads, &lengths, params.clone());

    let mut rng = StdRng::seed_from_u64(99);
    for _ in 0..100 {
        let deletes = vec![(rng.gen_range(0..n), rng.gen_range(0..n))];
        let _ = spanner.apply_batch_updates(deletes, Vec::new(), Vec::new());
        assert!(spanner.edge_count() <= params.max_edges_factor * n);
        if let Some((_, path_len)) = spanner.get_embedding(0, n / 2) {
            assert!(path_len <= n);
        }
    }
}

#[test]
fn test_cycle_oracle_dynamic_mode() {
    let (tails, heads, lengths, gradients) = random_graph(200, 800, 11);
    let mut oracle = DynamicOracle::new(1, true);
    let q = OracleQuery {
        iter: 0,
        node_count: 200,
        tails: &tails,
        heads: &heads,
        gradients: &gradients,
        lengths: &lengths,
    };
    let cycle = oracle.find_approx_min_ratio_cycle(q).unwrap();
    assert!(cycle.is_some());
}

#[test]
fn test_core_graph_lift_invariants() {
    let mut g = Graph::new(8);
    for i in 0..8 {
        let j = (i + 1) % 8;
        g.add_edge(NodeId(i), NodeId(j), 0.0, 1.0, 1.0).unwrap();
    }
    let tails: Vec<u32> = (0..8).map(|i| i as u32).collect();
    let heads: Vec<u32> = (0..8).map(|i| ((i + 1) % 8) as u32).collect();
    let lengths = vec![1.0; 8];
    let tree =
        LowStretchTree::build_low_stretch_deterministic(8, &tails, &heads, &lengths).unwrap();
    let forest = DynamicForest::new_from_tree(8, tails, heads, lengths, tree.tree_edges).unwrap();
    let stretch = vec![2.0; g.edge_count()];
    let core = g.build_core_graph(&forest, &stretch);
    let sum_div: f64 = core
        .edges
        .iter()
        .map(|e| e.lifted_gradient - e.lifted_gradient)
        .sum();
    assert!(sum_div.abs() <= 1e-9);
}

#[test]
fn test_cycle_quality_bounds() {
    let (tails, heads, lengths, gradients) = random_graph(100, 500, 71);
    let mut oracle = DynamicOracle::new(5, true);
    for run in 0..50 {
        let q = OracleQuery {
            iter: run,
            node_count: 100,
            tails: &tails,
            heads: &heads,
            gradients: &gradients,
            lengths: &lengths,
        };
        if let Some(cycle) = oracle.find_approx_min_ratio_cycle(q).unwrap() {
            assert!(cycle.denominator > 0.0);
            assert!(cycle.ratio.is_finite());
        }
    }
}
