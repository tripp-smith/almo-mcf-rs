use rand::prelude::*;

use almo_mcf_core::graph::{Graph, NodeId};
use almo_mcf_core::spanner::{build_spanner, contract_with_spanner, SpannerConfig};
use almo_mcf_core::trees::forest::DynamicForest;
use almo_mcf_core::trees::LowStretchTree;

fn build_random_graph(node_count: usize, p: f64, seed: u64) -> Graph {
    let mut graph = Graph::new(node_count);
    let mut rng = StdRng::seed_from_u64(seed);
    for i in 0..node_count {
        for j in (i + 1)..node_count {
            if rng.gen::<f64>() <= p {
                let _ = graph.add_edge(NodeId(i), NodeId(j), 0.0, 1.0, 1.0);
                let _ = graph.add_edge(NodeId(j), NodeId(i), 0.0, 1.0, 1.0);
            }
        }
    }
    graph
}

fn bfs_distance(graph: &Graph, start: usize, end: usize) -> Option<usize> {
    if start == end {
        return Some(0);
    }
    let mut queue = std::collections::VecDeque::new();
    let mut seen = vec![false; graph.node_count()];
    queue.push_back(start);
    seen[start] = true;
    let mut dist = vec![0usize; graph.node_count()];
    while let Some(node) = queue.pop_front() {
        if node == end {
            return Some(dist[node]);
        }
        let outgoing = graph.outgoing_edges(NodeId(node)).ok()?;
        for edge_id in outgoing {
            let edge = graph.edge(edge_id)?;
            let next = edge.head.0;
            if !seen[next] {
                seen[next] = true;
                dist[next] = dist[node] + 1;
                queue.push_back(next);
            }
        }
        let incoming = graph.incoming_edges(NodeId(node)).ok()?;
        for edge_id in incoming {
            let edge = graph.edge(edge_id)?;
            let next = edge.tail.0;
            if !seen[next] {
                seen[next] = true;
                dist[next] = dist[node] + 1;
                queue.push_back(next);
            }
        }
    }
    None
}

#[test]
fn spanner_tracks_size_and_stretch_under_deletions() {
    let node_count = 120;
    let graph = build_random_graph(node_count, 1.0 / (node_count as f64).sqrt(), 7);
    let config = SpannerConfig::from_graph(node_count, graph.edge_count(), None);
    let mut spanner = build_spanner(&graph, &config);
    assert!(spanner.edge_count() <= config.size_bound);

    let mut rng = StdRng::seed_from_u64(11);
    let mut total_stretch = 0.0;
    let mut samples = 0;
    for _ in 0..100 {
        let s = rng.gen_range(0..node_count);
        let t = rng.gen_range(0..node_count);
        if let Some(dist) = bfs_distance(&graph, s, t) {
            if dist == 0 {
                continue;
            }
            if let Some(metrics) = spanner.get_embedding_metrics(s, t) {
                let stretch = (metrics.1 as f64) / (dist as f64);
                total_stretch += stretch;
                samples += 1;
            }
        }
    }
    if samples > 0 {
        let avg_stretch = total_stretch / samples as f64;
        assert!(avg_stretch <= config.stretch.max(1.0));
    }

    let mut deletes = Vec::new();
    for (edge_id, _edge) in graph.edges().take(40) {
        let edge_ref = graph.edge(edge_id).unwrap();
        deletes.push(edge_ref.clone());
    }
    let updates = spanner.batch_update(&[], &deletes);
    let limit = config.recourse_limit(node_count);
    assert!(updates.recourse <= limit + 5);
}

#[test]
fn spanner_integration_contracts_core() {
    let node_count = 20;
    let graph = build_random_graph(node_count, 0.4, 13);
    let config = SpannerConfig::from_graph(node_count, graph.edge_count(), None);
    let spanner = build_spanner(&graph, &config);

    let mut tails = Vec::new();
    let mut heads = Vec::new();
    let mut lengths = Vec::new();
    for (edge_id, edge) in graph.edges() {
        let _ = edge_id;
        tails.push(edge.tail.0 as u32);
        heads.push(edge.head.0 as u32);
        lengths.push(1.0);
    }
    let tree =
        LowStretchTree::build_low_stretch_deterministic(node_count, &tails, &heads, &lengths)
            .expect("tree build");
    let forest = DynamicForest::new_from_tree(node_count, tails, heads, lengths, tree.tree_edges)
        .expect("forest build");
    let core = contract_with_spanner(&graph, &forest, &spanner);
    assert!(core.node_count >= 1);
    assert!(core.tails.len() <= node_count);
}
