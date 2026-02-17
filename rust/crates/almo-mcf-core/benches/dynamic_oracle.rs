use almo_mcf_core::{min_cost_flow_exact, McfOptions, McfProblem, Strategy};
use criterion::{criterion_group, criterion_main, Criterion};

fn build_problem(n: usize, m: usize) -> McfProblem {
    let mut tails = Vec::with_capacity(m);
    let mut heads = Vec::with_capacity(m);
    let mut lower = Vec::with_capacity(m);
    let mut upper = Vec::with_capacity(m);
    let mut cost = Vec::with_capacity(m);
    for i in 0..m {
        let u = i % n;
        let v = (i * 17 + 13) % n;
        let v = if v == u { (v + 1) % n } else { v };
        tails.push(u as u32);
        heads.push(v as u32);
        lower.push(0);
        upper.push(10);
        cost.push((i % 11) as i64 - 5);
    }
    let mut demand = vec![0_i64; n];
    demand[0] = -100;
    demand[n - 1] = 100;
    McfProblem::new(tails, heads, lower, upper, cost, demand).expect("valid")
}

fn bench_dynamic_oracle(c: &mut Criterion) {
    let problem = build_problem(1_000, 10_000);
    let opts = McfOptions {
        strategy: Strategy::FullDynamic,
        use_ipm: Some(true),
        max_iters: 100,
        ..McfOptions::default()
    };
    c.bench_function("dynamic_oracle_full_chain", |b| {
        b.iter(|| {
            let _ = min_cost_flow_exact(&problem, &opts);
        })
    });
}

criterion_group!(benches, bench_dynamic_oracle);
criterion_main!(benches);
