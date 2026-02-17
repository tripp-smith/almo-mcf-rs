use almo_mcf_core::graph::{Graph, NodeId};
use almo_mcf_core::min_ratio::oracle::DynamicOracle;
use almo_mcf_core::min_ratio::OracleQuery;

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
fn test_oracle_rebuilding_integration() {
    let n = 300usize;
    let m = 1200usize;
    let g = make_graph(n, m);
    let tails: Vec<u32> = g.edges().map(|(_, e)| e.tail.0 as u32).collect();
    let heads: Vec<u32> = g.edges().map(|(_, e)| e.head.0 as u32).collect();
    let lengths: Vec<f64> = g.edges().map(|(_, e)| e.cost.max(1.0)).collect();
    let gradients = vec![0.1; tails.len()];
    let mut oracle = DynamicOracle::new(0, true, Some(42));
    let mut found = 0usize;
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
            found = found.saturating_add(1);
        }
    }
    assert!(found >= 190);
}
