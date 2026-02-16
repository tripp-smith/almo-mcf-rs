use almo_mcf_core::{run_with_rebuilding_bench, McfOptions, McfProblem};
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_rebuilding_amortized(c: &mut Criterion) {
    c.bench_function("rebuilding_amortized", |b| {
        b.iter(|| {
            let n = 60usize;
            let m = 240usize;
            let mut tails = Vec::new();
            let mut heads = Vec::new();
            let mut lower = Vec::new();
            let mut upper = Vec::new();
            let mut cost = Vec::new();
            for i in 0..m {
                tails.push((i % n) as u32);
                heads.push(((i * 11 + 5) % n) as u32);
                lower.push(0);
                upper.push(20);
                cost.push(((i % 7) + 1) as i64);
            }
            let mut demands = vec![0i64; n];
            demands[0] = -20;
            demands[2] = 20;
            let problem = McfProblem::new(tails, heads, lower, upper, cost, demands).unwrap();
            let _ = run_with_rebuilding_bench(&problem, &McfOptions::default()).unwrap();
        })
    });
}

criterion_group!(benches, bench_rebuilding_amortized);
criterion_main!(benches);
