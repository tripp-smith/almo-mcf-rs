use almo_mcf_core::ipm::run_ipm;
use almo_mcf_core::solver::set_dynamic_max_iters;
use almo_mcf_core::{McfOptions, McfProblem, OracleMode, Strategy};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[test]
fn test_iteration_optimization() {
    let m = 1500;
    let iters = set_dynamic_max_iters(m, 0.2);
    assert!(iters > 0);
    let bound = ((m as f64).sqrt() * (m as f64).ln().max(1.0) * 10.0) as usize;
    assert!(iters <= bound.max(1_000_000));
}

fn build_feasible_problem(rng: &mut StdRng, n: usize, m: usize) -> McfProblem {
    let mut tails = Vec::with_capacity(m);
    let mut heads = Vec::with_capacity(m);
    let mut lower = Vec::with_capacity(m);
    let mut upper = Vec::with_capacity(m);
    let mut cost = Vec::with_capacity(m);
    let mut flow = Vec::with_capacity(m);

    for _ in 0..m {
        let tail = rng.gen_range(0..n) as u32;
        let mut head = rng.gen_range(0..n) as u32;
        if head == tail {
            head = (head + 1) % n as u32;
        }
        let up = rng.gen_range(5..=30) as i64;
        let f = rng.gen_range(0..=up);
        tails.push(tail);
        heads.push(head);
        lower.push(0);
        upper.push(up);
        cost.push(rng.gen_range(-3..=5) as i64);
        flow.push(f);
    }

    let mut demands = vec![0_i64; n];
    for (idx, &f) in flow.iter().enumerate() {
        let tail = tails[idx] as usize;
        let head = heads[idx] as usize;
        demands[tail] -= f;
        demands[head] += f;
    }

    McfProblem::new(tails, heads, lower, upper, cost, demands).unwrap()
}

#[test]
fn test_ipm_convergence_tuning() {
    let mut rng = StdRng::seed_from_u64(7);
    let problem = build_feasible_problem(&mut rng, 30, 100);
    let opts = McfOptions {
        max_iters: 200,
        tolerance: 1e-9,
        use_ipm: Some(true),
        oracle_mode: OracleMode::Fallback,
        strategy: Strategy::PeriodicRebuild { rebuild_every: 1 },
        ..McfOptions::default()
    };
    let result = run_ipm(&problem, &opts).unwrap();
    assert!(
        result.stats.iterations < 50,
        "iters={}",
        result.stats.iterations
    );
    assert!(result.stats.last_gap.is_finite());
}
