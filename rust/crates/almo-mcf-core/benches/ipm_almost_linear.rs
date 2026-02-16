use almo_mcf_core::solver::run_full_ipm;
use almo_mcf_core::{McfOptions, McfProblem};
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_ipm_almost_linear(c: &mut Criterion) {
    let problem = McfProblem::new(
        vec![0, 0, 1, 2, 3],
        vec![1, 2, 2, 3, 4],
        vec![0, 0, 0, 0, 0],
        vec![10, 10, 10, 10, 10],
        vec![1, 2, 1, 2, 1],
        vec![-5, 0, 0, 0, 5],
    )
    .unwrap();
    c.bench_function("ipm_almost_linear", |b| {
        b.iter(|| run_full_ipm(&problem, &McfOptions::default()).unwrap())
    });
}

criterion_group!(benches, bench_ipm_almost_linear);
criterion_main!(benches);
