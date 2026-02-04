use almo_mcf_core::min_ratio::oracle::DynamicOracle;
use almo_mcf_core::min_ratio::{MinRatioOracle, OracleQuery};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_graph(seed: u64, node_count: usize, edge_count: usize) -> (Vec<u32>, Vec<u32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut tails = Vec::with_capacity(edge_count);
    let mut heads = Vec::with_capacity(edge_count);
    for _ in 0..edge_count {
        let mut tail = rng.gen_range(0..node_count) as u32;
        let mut head = rng.gen_range(0..node_count) as u32;
        while head == tail {
            head = rng.gen_range(0..node_count) as u32;
        }
        if head < tail {
            std::mem::swap(&mut tail, &mut head);
        }
        tails.push(tail);
        heads.push(head);
    }
    (tails, heads)
}

fn random_values(seed: u64, edge_count: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut lengths = Vec::with_capacity(edge_count);
    let mut gradients = Vec::with_capacity(edge_count);
    for _ in 0..edge_count {
        lengths.push(rng.gen_range(0.5..5.0));
        gradients.push(rng.gen_range(-2.0..2.0));
    }
    (lengths, gradients)
}

#[test]
fn dynamic_oracle_matches_baseline_ratio() {
    let node_count = 50;
    let edge_count = 200;
    let (tails, heads) = random_graph(42, node_count, edge_count);
    let (lengths, gradients) = random_values(7, edge_count);

    let mut dynamic = DynamicOracle::new(1, true);
    let mut baseline = MinRatioOracle::new_with_mode(1, 1, true, None);

    let query = OracleQuery {
        iter: 0,
        node_count,
        tails: &tails,
        heads: &heads,
        gradients: &gradients,
        lengths: &lengths,
    };

    let dynamic_best = dynamic.find_approx_min_ratio_cycle(query).unwrap();
    let baseline_best = baseline.best_cycle(query).unwrap();
    match (dynamic_best, baseline_best) {
        (Some(dynamic_cycle), Some(baseline_cycle)) => {
            let diff = (dynamic_cycle.ratio - baseline_cycle.ratio).abs();
            assert!(diff <= 1e-9);
        }
        (None, None) => {}
        _ => panic!("oracle mismatch"),
    }
}

#[test]
fn dynamic_oracle_handles_updates() {
    let node_count = 30;
    let edge_count = 120;
    let (tails, heads) = random_graph(123, node_count, edge_count);
    let (mut lengths, mut gradients) = random_values(55, edge_count);
    let mut dynamic = DynamicOracle::new(3, true);

    let query = OracleQuery {
        iter: 0,
        node_count,
        tails: &tails,
        heads: &heads,
        gradients: &gradients,
        lengths: &lengths,
    };
    assert!(dynamic
        .find_approx_min_ratio_cycle(query)
        .unwrap()
        .is_some());

    for idx in (0..edge_count).step_by(7) {
        lengths[idx] *= 1.5;
        gradients[idx] -= 0.25;
    }

    let query = OracleQuery {
        iter: 1,
        node_count,
        tails: &tails,
        heads: &heads,
        gradients: &gradients,
        lengths: &lengths,
    };
    assert!(dynamic
        .find_approx_min_ratio_cycle(query)
        .unwrap()
        .is_some());
}
