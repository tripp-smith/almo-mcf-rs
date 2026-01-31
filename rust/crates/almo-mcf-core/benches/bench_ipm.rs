use almo_mcf_core::{min_cost_flow_exact, McfOptions, McfProblem};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[derive(Clone)]
struct BenchRng(u64);

impl BenchRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max
    }

    fn next_i64(&mut self, min: i64, max: i64) -> i64 {
        let span = (max - min + 1) as u64;
        min + (self.next_u64() % span) as i64
    }
}

fn build_grid_problem(rows: usize, cols: usize) -> McfProblem {
    let node_count = rows * cols;
    let mut tails = Vec::new();
    let mut heads = Vec::new();
    let mut upper = Vec::new();
    let mut cost = Vec::new();
    let mut rng = BenchRng::new(17);

    let idx = |r: usize, c: usize| -> usize { r * cols + c };
    for r in 0..rows {
        for c in 0..cols {
            let node = idx(r, c);
            if c + 1 < cols {
                tails.push(node as u32);
                heads.push(idx(r, c + 1) as u32);
                upper.push(8);
                cost.push(rng.next_i64(0, 4));
            }
            if r + 1 < rows {
                tails.push(node as u32);
                heads.push(idx(r + 1, c) as u32);
                upper.push(8);
                cost.push(rng.next_i64(-1, 5));
            }
            if r + 1 < rows && c + 1 < cols {
                tails.push(node as u32);
                heads.push(idx(r + 1, c + 1) as u32);
                upper.push(5);
                cost.push(rng.next_i64(-2, 6));
            }
        }
    }

    let edge_count = tails.len();
    let lower = vec![0_i64; edge_count];
    let mut demands = vec![0_i64; node_count];
    demands[0] = -20;
    demands[node_count - 1] = 20;

    McfProblem::new(tails, heads, lower, upper, cost, demands).expect("valid grid problem")
}

fn build_bipartite_problem(left: usize, right: usize, edges_per_left: usize) -> McfProblem {
    let node_count = left + right;
    let mut tails = Vec::new();
    let mut heads = Vec::new();
    let mut upper = Vec::new();
    let mut cost = Vec::new();
    let mut rng = BenchRng::new(23);

    for l in 0..left {
        for _ in 0..edges_per_left {
            let r = rng.next_usize(right);
            tails.push(l as u32);
            heads.push((left + r) as u32);
            upper.push(2);
            cost.push(rng.next_i64(-3, 7));
        }
    }

    let edge_count = tails.len();
    let lower = vec![0_i64; edge_count];
    let mut demands = vec![0_i64; node_count];
    for demand in demands.iter_mut().take(left) {
        *demand = -1;
    }
    for demand in demands.iter_mut().skip(left) {
        *demand = 1;
    }

    McfProblem::new(tails, heads, lower, upper, cost, demands).expect("valid bipartite problem")
}

fn build_random_problem(node_count: usize, edge_count: usize) -> McfProblem {
    let mut rng = BenchRng::new(42);
    let mut tails = Vec::with_capacity(edge_count);
    let mut heads = Vec::with_capacity(edge_count);
    let lower = vec![0_i64; edge_count];
    let mut upper = Vec::with_capacity(edge_count);
    let mut cost = Vec::with_capacity(edge_count);

    for _ in 0..edge_count {
        let tail = rng.next_usize(node_count);
        let mut head = rng.next_usize(node_count);
        if head == tail {
            head = (head + 1) % node_count;
        }
        tails.push(tail as u32);
        heads.push(head as u32);
        upper.push(20);
        cost.push(rng.next_i64(-10, 10));
    }

    let mut demands = vec![0_i64; node_count];
    demands[0] = -15;
    demands[node_count - 1] = 15;

    McfProblem::new(tails, heads, lower, upper, cost, demands).expect("valid random problem")
}

fn bench_min_cost_flow(c: &mut Criterion) {
    let mut group = c.benchmark_group("min_cost_flow_exact");
    let grid = build_grid_problem(12, 12);
    group.throughput(Throughput::Elements(grid.edge_count() as u64));
    group.bench_with_input(BenchmarkId::new("grid", "12x12"), &grid, |b, problem| {
        b.iter(|| {
            let _ = min_cost_flow_exact(problem, &McfOptions::default());
        })
    });

    let bipartite = build_bipartite_problem(40, 40, 8);
    group.throughput(Throughput::Elements(bipartite.edge_count() as u64));
    group.bench_with_input(
        BenchmarkId::new("bipartite", "40x40"),
        &bipartite,
        |b, problem| {
            b.iter(|| {
                let _ = min_cost_flow_exact(problem, &McfOptions::default());
            })
        },
    );

    let random = build_random_problem(120, 600);
    group.throughput(Throughput::Elements(random.edge_count() as u64));
    group.bench_with_input(
        BenchmarkId::new("random", "120x600"),
        &random,
        |b, problem| {
            b.iter(|| {
                let _ = min_cost_flow_exact(problem, &McfOptions::default());
            })
        },
    );
    group.finish();
}

criterion_group!(benches, bench_min_cost_flow);
criterion_main!(benches);
