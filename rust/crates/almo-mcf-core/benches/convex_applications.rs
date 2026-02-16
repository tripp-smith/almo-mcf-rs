use almo_mcf_core::convex::{entropy_regularized_ot, BipartiteGraph};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_entropy_ot(c: &mut Criterion) {
    let n = 100;
    let mut edges = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if (i + j) % 2 == 0 {
                edges.push((i, j));
            }
        }
    }
    let g = BipartiteGraph {
        left: n,
        right: n,
        edges: edges.clone(),
    };
    let costs = vec![1.0; edges.len()];
    let demands = vec![1.0; 2 * n];

    c.bench_function("entropy_ot", |b| {
        b.iter(|| {
            let _ =
                entropy_regularized_ot(black_box(&g), black_box(&demands), black_box(&costs), 0.5);
        })
    });
}

criterion_group!(benches, bench_entropy_ot);
criterion_main!(benches);
