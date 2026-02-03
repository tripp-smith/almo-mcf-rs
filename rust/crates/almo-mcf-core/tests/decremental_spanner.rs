use almo_mcf_core::spanner::{DecrementalSpanner, DecrementalSpannerParams};
use proptest::prelude::*;

fn build_complete_graph(n: usize) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let mut tails = Vec::new();
    let mut heads = Vec::new();
    let mut lengths = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            tails.push(u as u32);
            heads.push(v as u32);
            lengths.push(1.0);
        }
    }
    (tails, heads, lengths)
}

#[test]
fn spanner_quality_on_decremental_sequence() {
    let node_count = 8;
    let (tails, heads, lengths) = build_complete_graph(node_count);
    let params = DecrementalSpannerParams::from_graph(node_count, tails.len());
    let mut spanner = DecrementalSpanner::new(node_count, &tails, &heads, &lengths, params);

    let deletes = vec![(0, 1), (2, 3), (4, 5)];
    spanner
        .apply_batch_updates(deletes, Vec::new(), Vec::new())
        .expect("batch updates");

    assert!(spanner.edge_count() <= node_count * 8);
    let embedding = spanner.get_embedding(0, 2).expect("path exists");
    assert!(embedding.1 <= 8);
}

proptest! {
    #[test]
    fn prop_spanner_respects_edge_budget(delete_count in 0usize..5) {
        let node_count = 6;
        let (tails, heads, lengths) = build_complete_graph(node_count);
        let params = DecrementalSpannerParams::from_graph(node_count, tails.len());
        let mut spanner = DecrementalSpanner::new(node_count, &tails, &heads, &lengths, params);
        let mut deletes = Vec::new();
        for idx in 0..delete_count {
            deletes.push((idx % node_count, (idx + 1) % node_count));
        }
        spanner
            .apply_batch_updates(deletes, Vec::new(), Vec::new())
            .expect("batch updates");
        prop_assert!(spanner.edge_count() <= node_count * 8);
    }
}
