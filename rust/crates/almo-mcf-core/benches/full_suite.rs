use almo_mcf_core::{min_cost_flow_exact, min_cost_flow_scaled, McfOptions, McfProblem};
use criterion::{criterion_group, criterion_main, Criterion};

fn tiny_problem(m: usize) -> McfProblem {
    let n = (m / 5).max(4);
    let mut tails = Vec::new();
    let mut heads = Vec::new();
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    let mut cost = Vec::new();
    for i in 0..m {
        let u = i % n;
        let v = (i + 1) % n;
        tails.push(u as u32);
        heads.push(v as u32);
        lower.push(0);
        upper.push(20);
        cost.push(1);
    }
    let mut demand = vec![0_i64; n];
    demand[0] = -10;
    demand[n - 1] = 10;
    McfProblem::new(tails, heads, lower, upper, cost, demand).expect("valid")
}

fn bench_full_suite(c: &mut Criterion) {
    let p = tiny_problem(5_000);
    let opts = McfOptions::default();
    c.bench_function("full_suite_exact", |b| {
        b.iter(|| min_cost_flow_exact(&p, &opts))
    });
    c.bench_function("full_suite_scaled", |b| {
        b.iter(|| min_cost_flow_scaled(&p, &opts))
    });
}

criterion_group!(benches, bench_full_suite);
criterion_main!(benches);
