use almo_mcf_core::data_structures::chain::{ChainParams, Circulation, DataStructureChain};
use almo_mcf_core::data_structures::embedding::DynamicEmbedding;
use almo_mcf_core::data_structures::lsd::Path;
use almo_mcf_core::graph::{EdgeId, Graph, NodeId};
use almo_mcf_core::min_ratio::oracle::DynamicOracle;
use almo_mcf_core::min_ratio::OracleQuery;
use almo_mcf_core::rebuilding::RebuildingGame;

fn make_graph(n: usize, m: usize) -> Graph {
    let mut g = Graph::new(n);
    for e in 0..m {
        let u = e % n;
        let v = (e * 17 + 5) % n;
        if u != v {
            let _ = g.add_edge(NodeId(u), NodeId(v), 0.0, 1.0, 1.0 + (e % 13) as f64);
        }
    }
    g
}

#[test]
fn test_rebuilding_game_rounds() {
    let n = 512usize;
    let m = 2048usize;
    let mut game = RebuildingGame::new(n, m, 6);
    let mut rebuild_counts = vec![0usize; 6];
    for t in 1..=1000 {
        let is_win = t % 5 != 0;
        let d = game.play_round(t, 3, is_win);
        if let Some(level) = d.level {
            rebuild_counts[level] += 1;
        }
    }
    for count in rebuild_counts {
        assert!(count <= ((n as f64).ln().ceil() as usize).saturating_mul(3));
    }
    assert!(game.total_recourse <= (m as f64) * (n as f64).ln().powi(2) * 10.0);
}

#[test]
fn test_oracle_rebuilding_integration() {
    let n = 300usize;
    let m = 1200usize;
    let g = make_graph(n, m);
    let tails: Vec<u32> = g.edges().map(|(_, e)| e.tail.0 as u32).collect();
    let heads: Vec<u32> = g.edges().map(|(_, e)| e.head.0 as u32).collect();
    let lengths: Vec<f64> = g.edges().map(|(_, e)| e.cost.max(1.0)).collect();
    let gradients = vec![0.1; tails.len()];
    let mut oracle = DynamicOracle::new(0, true);
    let mut good = 0usize;
    for iter in 0..200 {
        let q = OracleQuery {
            iter,
            node_count: n,
            tails: &tails,
            heads: &heads,
            gradients: &gradients,
            lengths: &lengths,
        };
        if oracle.find_approx_min_ratio_cycle(q).unwrap().is_some() {
            good += 1;
        }
    }
    assert!(good >= 190);
}

#[test]
fn test_dynamic_embeddings() {
    let mut emb = DynamicEmbedding::new(4);
    let path = Path {
        edges: (0..20).map(EdgeId).collect(),
        stretch: (200f64).ln(),
    };
    let deletions: Vec<EdgeId> = (0..5).map(EdgeId).collect();
    for _ in 0..50 {
        let ep = emb.embed_dynamic_path(&path, 0, &deletions);
        assert!(ep.distortion <= (200f64).ln() * 4.0);
    }
    assert!(emb.amortize_update_cost(50) < (1200f64).powf(0.05) * 20.0);
}

#[test]
fn test_derandomized_reproducibility() {
    let mut a = RebuildingGame::new(400, 2000, 5);
    let mut b = RebuildingGame::new(400, 2000, 5);
    a.enable_derandomization(9);
    b.enable_derandomization(9);
    let mut da = Vec::new();
    let mut db = Vec::new();
    for t in 1..200 {
        da.push(a.play_round(t, 2, t % 3 != 0).level);
        db.push(b.play_round(t, 2, t % 3 != 0).level);
    }
    assert_eq!(da, db);
    assert!(a.verify_derandomized_invariants());
}

#[test]
fn test_amortized_guarantees() {
    let graph = make_graph(500, 2500);
    let mut chain = DataStructureChain::initialize_chain(&graph, ChainParams::default());
    let mut game = RebuildingGame::new(graph.node_count(), graph.edge_count(), 5);
    game.enable_derandomization(1);
    let circulation = Circulation {
        candidate_edges: (0..120).map(EdgeId).collect(),
    };
    for t in 1..=500 {
        let _ = chain.query_min_ratio_cycle_chain(&circulation, t, &mut game);
    }
    let total_time = game.compute_amortized_time(500) * 500.0;
    assert!(total_time <= (2500f64).powf(1.05) * 100.0);
    let stats = game.log_game_stats(500);
    let avg_win =
        stats.win_rate_per_level.iter().sum::<f64>() / stats.win_rate_per_level.len() as f64;
    assert!(avg_win >= 0.5);
}
