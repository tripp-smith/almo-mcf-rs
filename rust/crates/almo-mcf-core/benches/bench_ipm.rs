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
}

fn build_problem(node_count: usize, edge_count: usize) -> McfProblem {
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
        cost.push((rng.next_u64() % 21) as i64 - 10);
    }

    let mut demands = vec![0_i64; node_count];
    demands[0] = -15;
    demands[node_count - 1] = 15;

    McfProblem::new(tails, heads, lower, upper, cost, demands).expect("valid problem")
}

fn bench_min_cost_flow(c: &mut Criterion) {
    let mut group = c.benchmark_group("min_cost_flow_exact");
    let node_count = 100;
    let edge_count = 400;
    let problem = build_problem(node_count, edge_count);
    group.throughput(Throughput::Elements(problem.edge_count() as u64));
    group.bench_with_input(
        BenchmarkId::new("dense", edge_count),
        &problem,
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
