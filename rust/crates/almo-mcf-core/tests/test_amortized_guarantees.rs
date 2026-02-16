use almo_mcf_core::data_structures::chain::{ChainParams, Circulation, DataStructureChain};
use almo_mcf_core::graph::{EdgeId, Graph, NodeId};
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
    assert!(total_time.is_finite());
    let stats = game.log_game_stats(500);
    let avg_win =
        stats.win_rate_per_level.iter().sum::<f64>() / stats.win_rate_per_level.len() as f64;
    assert!(avg_win >= 0.5);
}
