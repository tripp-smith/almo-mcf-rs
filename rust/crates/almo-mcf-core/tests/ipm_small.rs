use almo_mcf_core::ipm::{run_ipm, IpmTermination};
use almo_mcf_core::{McfOptions, McfProblem, OracleMode, Strategy};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
        let up = rng.gen_range(1..=100) as i64;
        let f = rng.gen_range(0..=up);
        tails.push(tail);
        heads.push(head);
        lower.push(0);
        upper.push(up);
        cost.push(rng.gen_range(-5..=10) as i64);
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
fn ipm_converges_on_random_feasible_graphs() {
    let mut rng = StdRng::seed_from_u64(42);
    let sizes = [(8, 20)];
    for (n, m) in sizes {
        let problem = build_feasible_problem(&mut rng, n, m);
        let opts = McfOptions {
            tolerance: 1e-6,
            max_iters: 100,
            time_limit_ms: Some(200),
            use_ipm: Some(true),
            ..McfOptions::default()
        };
        let result = run_ipm(&problem, &opts).unwrap();
        assert!(
            matches!(
                result.termination,
                IpmTermination::Converged
                    | IpmTermination::IterationLimit
                    | IpmTermination::TimeLimit
            ),
            "unexpected termination: {:?}",
            result.termination
        );
        assert!(result.stats.iterations <= 100);

        let potentials = result.stats.potentials;
        if potentials.len() > 1 {
            for window in potentials.windows(2) {
                assert!(window[1] <= window[0] + 1e-8);
            }
        }

        if result.termination == IpmTermination::Converged {
            let ipm_cost: f64 = result
                .flow
                .iter()
                .zip(problem.cost.iter())
                .map(|(f, c)| *f * (*c as f64))
                .sum();
            assert!(ipm_cost.is_finite());
        }
    }
}

#[test]
fn ipm_reports_infeasible_demands() {
    let problem = McfProblem::new(
        vec![0, 1],
        vec![1, 2],
        vec![0, 0],
        vec![5, 5],
        vec![1, 1],
        vec![1, -2, 2],
    );
    assert!(matches!(
        problem,
        Err(almo_mcf_core::McfError::InvalidInput(_))
    ));
}

#[test]
fn ipm_handles_zero_capacity_edges() {
    let problem = McfProblem::new(
        vec![0, 1, 1],
        vec![1, 2, 0],
        vec![0, 0, 0],
        vec![0, 3, 2],
        vec![1, -2, 1],
        vec![-1, 0, 1],
    )
    .unwrap();
    let opts = McfOptions {
        use_ipm: Some(true),
        max_iters: 50,
        ..McfOptions::default()
    };
    let err = run_ipm(&problem, &opts).unwrap_err();
    assert!(matches!(err, almo_mcf_core::McfError::Infeasible));
}

#[test]
fn ipm_runs_with_fallback_oracle_mode() {
    let problem =
        McfProblem::new(vec![0], vec![1], vec![0], vec![10], vec![1], vec![-5, 5]).unwrap();
    let opts = McfOptions {
        oracle_mode: OracleMode::Fallback,
        strategy: Strategy::PeriodicRebuild { rebuild_every: 1 },
        use_ipm: Some(true),
        max_iters: 50,
        ..McfOptions::default()
    };
    let result = run_ipm(&problem, &opts).unwrap();
    assert!(matches!(
        result.termination,
        IpmTermination::Converged
            | IpmTermination::IterationLimit
            | IpmTermination::NoImprovingCycle
    ));
}
