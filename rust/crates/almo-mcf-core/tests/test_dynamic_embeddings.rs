use almo_mcf_core::data_structures::embedding::DynamicEmbedding;
use almo_mcf_core::data_structures::lsd::Path;
use almo_mcf_core::graph::EdgeId;

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
    assert!(emb.amortize_update_cost(50).is_finite());
}
