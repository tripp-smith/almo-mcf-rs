use almo_mcf_core::convex::{entropy_regularized_ot, isotonic_regression, BipartiteGraph, Dag};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_convex_extensions(c: &mut Criterion) {
    let n = 500;
    let mut edges = Vec::new();
    for i in 0..n {
        for j in 0..5 {
            edges.push((i, (i + j) % n));
        }
    }
    let graph = BipartiteGraph {
        left: n,
        right: n,
        edges: edges.clone(),
    };
    let demands = vec![1.0; 2 * n];
    let costs = vec![1.0; edges.len()];

    c.bench_function("convex_entropy_ot", |b| {
        b.iter(|| {
            let _ = entropy_regularized_ot(
                black_box(&graph),
                black_box(&demands),
                black_box(&costs),
                0.25,
            );
        })
    });

    let dag = Dag {
        n,
        edges: (0..n - 1).map(|i| (i, i + 1)).collect(),
    };
    let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
    c.bench_function("convex_isotonic", |b| {
        b.iter(|| {
            let _ = isotonic_regression(black_box(&dag), black_box(&y), 2.0);
        })
    });
}

criterion_group!(benches, bench_convex_extensions);
criterion_main!(benches);
