use almo_mcf_core::{min_cost_flow_exact, McfOptions, McfProblem, OracleMode, Strategy};

fn path_problem(n: usize) -> McfProblem {
    let mut tails = Vec::new();
    let mut heads = Vec::new();
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    let mut cost = Vec::new();
    for i in 0..(n - 1) {
        tails.push(i as u32);
        heads.push((i + 1) as u32);
        lower.push(0);
        upper.push(30);
        cost.push(1);
        tails.push((i + 1) as u32);
        heads.push(i as u32);
        lower.push(0);
        upper.push(30);
        cost.push(2);
    }
    let mut demands = vec![0i64; n];
    demands[0] = -10;
    demands[n - 1] = 10;
    McfProblem::new(tails, heads, lower, upper, cost, demands).unwrap()
}

#[test]
fn integration_ipm_dynamic() {
    for _ in 0..20 {
        let p = path_problem(40);
        let opts = McfOptions {
            strategy: Strategy::FullDynamic,
            oracle_mode: OracleMode::Dynamic,
            use_ipm: Some(true),
            use_scaling: Some(false),
            max_iters: 200,
            ..McfOptions::default()
        };
        let sol = min_cost_flow_exact(&p, &opts).unwrap();
        let stats = sol.ipm_stats.expect("ipm stats");
        assert!(matches!(stats.oracle_mode, OracleMode::Dynamic));
        assert!(stats.iterations <= opts.max_iters);
    }
}
