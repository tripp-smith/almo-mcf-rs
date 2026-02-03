use almo_mcf_core::spanner::{DecrementalSpanner, DecrementalSpannerParams};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn build_ring(n: usize) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let mut tails = Vec::with_capacity(n);
    let mut heads = Vec::with_capacity(n);
    let mut lengths = Vec::with_capacity(n);
    for i in 0..n {
        let u = i;
        let v = (i + 1) % n;
        tails.push(u as u32);
        heads.push(v as u32);
        lengths.push(1.0);
    }
    (tails, heads, lengths)
}

fn bench_build_and_query(c: &mut Criterion) {
    let sizes = [64usize, 128];
    for &size in &sizes {
        let (tails, heads, lengths) = build_ring(size);
        let mut params = DecrementalSpannerParams::from_graph(size, tails.len());
        let mut group = c.benchmark_group("decremental_spanner_build");
        for &deterministic in &[true, false] {
            params.deterministic = deterministic;
            group.bench_with_input(
                BenchmarkId::from_parameter(if deterministic { "det" } else { "rand" }),
                &size,
                |b, _| {
                    b.iter(|| {
                        DecrementalSpanner::new(size, &tails, &heads, &lengths, params.clone())
                    });
                },
            );
        }
        group.finish();

        let mut spanner = DecrementalSpanner::new(size, &tails, &heads, &lengths, params.clone());
        c.bench_with_input(
            BenchmarkId::new("decremental_spanner_query", size),
            &size,
            |b, _| {
                b.iter(|| spanner.get_embedding(0, size / 2));
            },
        );

        c.bench_with_input(
            BenchmarkId::new("decremental_spanner_update", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let _ = spanner.apply_batch_updates(vec![(0, 1)], Vec::new(), Vec::new());
                });
            },
        );
    }
}

criterion_group!(benches, bench_build_and_query);
criterion_main!(benches);
