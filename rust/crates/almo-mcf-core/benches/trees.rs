use almo_mcf_core::trees::LowStretchTree;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn build_chain_graph(node_count: usize) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let mut tails = Vec::new();
    let mut heads = Vec::new();
    let mut lengths = Vec::new();
    for i in 0..node_count - 1 {
        tails.push(i as u32);
        heads.push((i + 1) as u32);
        lengths.push(1.0 + (i as f64) * 0.01);
    }
    (tails, heads, lengths)
}

fn bench_tree_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_path_queries");
    for &size in &[64usize, 128, 256, 512] {
        let (tails, heads, lengths) = build_chain_graph(size);
        let tree = LowStretchTree::build_low_stretch(size, &tails, &heads, &lengths, 17).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                for i in 0..size / 2 {
                    let _ = tree.path_length(i, size - 1 - i);
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_tree_queries);
criterion_main!(benches);
