use almo_mcf_core::ipm;
use almo_mcf_core::{min_cost_flow_exact, McfOptions, McfProblem, Strategy};

#[derive(Clone)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            0
        } else {
            (self.next_u64() as usize) % max
        }
    }
}

fn grid_graph(side: usize) -> McfProblem {
    let node_count = side * side;
    let mut tails = Vec::new();
    let mut heads = Vec::new();
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    let mut cost = Vec::new();
    for r in 0..side {
        for c in 0..side {
            let node = r * side + c;
            if c + 1 < side {
                let right = node + 1;
                tails.push(node as u32);
                heads.push(right as u32);
                lower.push(0);
                upper.push(5);
                cost.push(1);
                tails.push(right as u32);
                heads.push(node as u32);
                lower.push(0);
                upper.push(5);
                cost.push(1);
            }
            if r + 1 < side {
                let down = node + side;
                tails.push(node as u32);
                heads.push(down as u32);
                lower.push(0);
                upper.push(5);
                cost.push(2);
                tails.push(down as u32);
                heads.push(node as u32);
                lower.push(0);
                upper.push(5);
                cost.push(2);
            }
        }
    }
    let mut demands = vec![0_i64; node_count];
    demands[0] = -6;
    demands[node_count - 1] = 6;
    McfProblem::new(tails, heads, lower, upper, cost, demands).unwrap()
}

fn transportation_graph() -> McfProblem {
    let supplies = [4_i64, 3];
    let demands = [3_i64, 4];
    let supply_count = supplies.len();
    let demand_count = demands.len();
    let node_count = supply_count + demand_count;
    let mut tails = Vec::new();
    let mut heads = Vec::new();
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    let mut cost = Vec::new();
    for (s_idx, _) in supplies.iter().enumerate() {
        for (d_idx, _) in demands.iter().enumerate() {
            tails.push(s_idx as u32);
            heads.push((supply_count + d_idx) as u32);
            lower.push(0);
            upper.push(10);
            cost.push((s_idx + d_idx) as i64 + 1);
        }
    }
    let mut node_demands = vec![0_i64; node_count];
    for (idx, supply) in supplies.iter().enumerate() {
        node_demands[idx] = -supply;
    }
    for (idx, demand) in demands.iter().enumerate() {
        node_demands[supply_count + idx] = *demand;
    }
    McfProblem::new(tails, heads, lower, upper, cost, node_demands).unwrap()
}

fn random_graph(seed: u64, node_count: usize, edge_count: usize) -> McfProblem {
    let mut rng = DeterministicRng::new(seed);
    let mut tails = Vec::new();
    let mut heads = Vec::new();
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    let mut cost = Vec::new();
    for _ in 0..edge_count {
        let u = rng.next_usize(node_count);
        let mut v = rng.next_usize(node_count);
        if u == v {
            v = (v + 1) % node_count;
        }
        tails.push(u as u32);
        heads.push(v as u32);
        lower.push(0);
        upper.push((rng.next_u64() % 5 + 1) as i64);
        cost.push((rng.next_u64() % 9) as i64 - 4);
    }
    tails.push(0);
    heads.push((node_count - 1) as u32);
    lower.push(0);
    upper.push(10);
    cost.push(5);
    let mut demands = vec![0_i64; node_count];
    demands[0] = -5;
    demands[node_count - 1] = 5;
    McfProblem::new(tails, heads, lower, upper, cost, demands).unwrap()
}

fn scale_free_graph(seed: u64, node_count: usize) -> McfProblem {
    let mut rng = DeterministicRng::new(seed);
    let mut tails = Vec::new();
    let mut heads = Vec::new();
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    let mut cost = Vec::new();
    let mut degrees = vec![1usize; node_count];
    for node in 1..node_count {
        let target = rng.next_usize(node);
        tails.push(node as u32);
        heads.push(target as u32);
        lower.push(0);
        upper.push(4);
        cost.push((node % 5) as i64);
        tails.push(target as u32);
        heads.push(node as u32);
        lower.push(0);
        upper.push(4);
        cost.push((node % 5) as i64);
        degrees[node] += 1;
        degrees[target] += 1;
        let extra = degrees[target].min(node_count - 1);
        let extra_target = (target + extra) % node_count;
        if extra_target != node {
            tails.push(node as u32);
            heads.push(extra_target as u32);
            lower.push(0);
            upper.push(3);
            cost.push(((node + extra_target) % 7) as i64 - 3);
        }
    }
    tails.push(0);
    heads.push((node_count - 1) as u32);
    lower.push(0);
    upper.push(10);
    cost.push(5);
    let mut demands = vec![0_i64; node_count];
    demands[0] = -6;
    demands[node_count - 1] = 6;
    McfProblem::new(tails, heads, lower, upper, cost, demands).unwrap()
}

fn solve(
    problem: &McfProblem,
    deterministic: bool,
    deterministic_seed: Option<u64>,
) -> (Vec<i64>, i128, usize) {
    let opts = McfOptions {
        deterministic,
        deterministic_seed,
        seed: 42,
        max_iters: 15,
        tolerance: 1e-9,
        use_ipm: Some(false),
        strategy: Strategy::PeriodicRebuild { rebuild_every: 5 },
        ..McfOptions::default()
    };
    let solution = min_cost_flow_exact(problem, &opts).unwrap();
    let iterations = solution
        .ipm_stats
        .as_ref()
        .map(|stats| stats.iterations)
        .unwrap_or(0);
    (solution.flow, solution.cost, iterations)
}

fn tiny_cycle_problem() -> McfProblem {
    McfProblem::new(
        vec![0, 1, 2],
        vec![1, 2, 0],
        vec![0, 0, 0],
        vec![4, 4, 4],
        vec![1, -2, 1],
        vec![0, 0, 0],
    )
    .unwrap()
}

#[test]
fn deterministic_runs_reproduce_exact_flows() {
    let graphs = [
        grid_graph(2),
        transportation_graph(),
        random_graph(0x5eed, 6, 10),
        scale_free_graph(0x1234, 5),
    ];
    for graph in &graphs {
        let (flow_a, cost_a, iters_a) = solve(graph, true, None);
        let (flow_b, cost_b, iters_b) = solve(graph, true, None);
        assert_eq!(flow_a, flow_b);
        assert_eq!(cost_a, cost_b);
        assert_eq!(iters_a, iters_b);

        let (flow_seed_a, cost_seed_a, iters_seed_a) = solve(graph, true, Some(17));
        let (flow_seed_b, cost_seed_b, iters_seed_b) = solve(graph, true, Some(17));
        assert_eq!(flow_seed_a, flow_seed_b);
        assert_eq!(cost_seed_a, cost_seed_b);
        assert_eq!(iters_seed_a, iters_seed_b);
    }
}

#[test]
fn deterministic_vs_randomized_costs_match() {
    let graph = grid_graph(3);
    let (_, cost_det, _) = solve(&graph, true, None);
    let (_, cost_rand, _) = solve(&graph, false, None);
    assert_eq!(cost_det, cost_rand);
}

#[test]
fn deterministic_long_run_is_stable() {
    let graph = tiny_cycle_problem();
    let opts = McfOptions {
        deterministic: true,
        deterministic_seed: Some(9),
        seed: 0,
        max_iters: 12,
        tolerance: 1e-9,
        use_ipm: Some(true),
        strategy: Strategy::PeriodicRebuild { rebuild_every: 4 },
        ..McfOptions::default()
    };
    let first = ipm::run_ipm(&graph, &opts).unwrap();
    let second = ipm::run_ipm(&graph, &opts).unwrap();
    assert_eq!(first.stats.iterations, second.stats.iterations);
    for (a, b) in first.flow.iter().zip(second.flow.iter()) {
        assert!((a - b).abs() <= 1e-9);
    }
}

#[test]
fn deterministic_golden_flow_matches_expected() {
    let problem = McfProblem::new(
        vec![0, 1, 0],
        vec![1, 2, 2],
        vec![0, 0, 0],
        vec![5, 5, 5],
        vec![1, 1, 3],
        vec![-5, 0, 5],
    )
    .unwrap();
    let opts = McfOptions {
        deterministic: true,
        deterministic_seed: None,
        seed: 0,
        use_ipm: Some(true),
        ..McfOptions::default()
    };
    let solution = min_cost_flow_exact(&problem, &opts).unwrap();
    assert_eq!(solution.flow, vec![5, 5, 0]);
    assert_eq!(solution.cost, 10);
}
