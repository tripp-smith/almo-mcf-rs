use almo_mcf_core::data_structures::chain::{ChainParams, Circulation, DataStructureChain};
use almo_mcf_core::data_structures::lsd::{LowStretchDecomposition, Path};
use almo_mcf_core::data_structures::sparsified_core::SparsifiedCoreGraph;
use almo_mcf_core::graph::{EdgeId, Graph, NodeId};
use almo_mcf_core::rebuilding::RebuildingGame;
use almo_mcf_core::trees::forest::DynamicForest;

fn make_graph(n: usize, m: usize) -> Graph {
    let mut g = Graph::new(n);
    for e in 0..m {
        let u = e % n;
        let v = (e * 7 + 3) % n;
        if u != v {
            let _ = g.add_edge(NodeId(u), NodeId(v), 0.0, 1.0, 1.0 + ((e % 5) as f64));
        }
    }
    g
}

#[test]
fn test_lsd_update() {
    let n = 150;
    let m = 600;
    let graph = make_graph(n, m);
    let mut lsd = LowStretchDecomposition::new(n, graph.edge_count(), 2.0);
    lsd.bootstrap_from_graph(&graph);
    let deletions: Vec<EdgeId> = (0..50.min(graph.edge_count())).map(EdgeId).collect();
    let updates = lsd.update_decomposition(&deletions, 16);
    assert!(!updates.is_empty());
    let avg_stretch = lsd.compute_average_stretch(&[Path {
        edges: deletions,
        stretch: (n as f64).ln().powi(2),
    }]);
    assert!(avg_stretch <= (n as f64).ln().powi(2) * 2.0);
}

#[test]
fn test_mw_stretch_regret() {
    let n = 200;
    let mut lsd = LowStretchDecomposition::new(n, 800, 2.0);
    let paths: Vec<Path> = (0..200)
        .map(|idx| Path {
            edges: vec![EdgeId(idx % 50)],
            stretch: ((idx + 1) as f64).ln().powi(2),
        })
        .collect();
    let weighted = lsd.apply_multiplicative_weights(&paths, &vec![1.0; 200]);
    assert!(weighted.average_stretch <= (n as f64).ln().powi(3));
    assert!(lsd.multiplicative_weights.regret_bound.is_finite());
}

#[test]
fn test_sparsified_core_embeddings() {
    let graph = make_graph(100, 400);
    let tails: Vec<u32> = graph.edges().map(|(_, e)| e.tail.0 as u32).collect();
    let heads: Vec<u32> = graph.edges().map(|(_, e)| e.head.0 as u32).collect();
    let lengths: Vec<f64> = graph.edges().map(|(_, e)| e.cost.abs().max(1.0)).collect();
    let tree_edges = vec![false; tails.len()];
    let forest = DynamicForest::new_from_tree(100, tails, heads, lengths, tree_edges)
        .expect("forest should build");
    let stretch = vec![2.0; graph.edge_count()];
    let core = graph.build_core_graph(&forest, &stretch);
    let lsd = LowStretchDecomposition::new(100, graph.edge_count(), 2.0);
    let gamma_l = (100f64).ln();
    let sparse = SparsifiedCoreGraph::sparsify_core_graph(&core, &lsd, gamma_l);
    let good = sparse
        .core_edges
        .iter()
        .filter(|se| {
            let embedded = sparse.embed_path(se.edge_id, 0);
            embedded.length <= gamma_l * se.length.max(1.0) * 4.0
        })
        .count();
    if !sparse.core_edges.is_empty() {
        assert!((good as f64) / (sparse.core_edges.len() as f64) >= 0.95);
    }
}

#[test]
fn test_full_chain_query() {
    let graph = make_graph(250, 1000);
    let mut chain = DataStructureChain::initialize_chain(&graph, ChainParams::default());
    let circulation = Circulation {
        candidate_edges: (0..100).map(EdgeId).collect(),
    };
    let mut game = RebuildingGame::new(graph.node_count(), graph.edge_count(), 3);
    let (cycle, quality) = chain.query_min_ratio_cycle_chain(&circulation, 32, &mut game);
    assert_eq!(cycle.edges.len(), circulation.candidate_edges.len());
    assert!(quality.is_finite());
}

#[test]
fn test_deterministic_chain_reproducibility() {
    let graph = make_graph(30, 120);
    let mut chain_a = DataStructureChain::initialize_chain(&graph, ChainParams::default());
    let mut chain_b = DataStructureChain::initialize_chain(&graph, ChainParams::default());
    let mut game_a = RebuildingGame::new(graph.node_count(), graph.edge_count(), 3);
    let mut game_b = RebuildingGame::new(graph.node_count(), graph.edge_count(), 3);
    assert!(chain_a.set_deterministic_mode(Some(7), &mut game_a));
    assert!(chain_b.set_deterministic_mode(Some(7), &mut game_b));
    let circulation = Circulation {
        candidate_edges: vec![EdgeId(0), EdgeId(1), EdgeId(2)],
    };
    let (ca, qa) = chain_a.query_min_ratio_cycle_chain(&circulation, 5, &mut game_a);
    let (cb, qb) = chain_b.query_min_ratio_cycle_chain(&circulation, 5, &mut game_b);
    assert_eq!(ca.edges, cb.edges);
    assert_eq!(qa, qb);
    let sa = chain_a.log_chain_stats(5);
    let sb = chain_b.log_chain_stats(5);
    assert_eq!(sa.seed_used, sb.seed_used);
}
