use almo_mcf_core::min_ratio::branching_tree_chain::{exact_best_cycle, BranchingTreeChain};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_graph(n: usize, m: usize, rng: &mut StdRng) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let mut tails = Vec::with_capacity(m);
    let mut heads = Vec::with_capacity(m);
    let mut lengths = Vec::with_capacity(m);
    for _ in 0..m {
        let u = rng.gen_range(0..n) as u32;
        let mut v = rng.gen_range(0..n) as u32;
        if v == u {
            v = (v + 1) % n as u32;
        }
        tails.push(u);
        heads.push(v);
        lengths.push(rng.gen_range(0.5..5.0));
    }
    (tails, heads, lengths)
}

#[test]
fn min_ratio_cycle_is_reasonable() {
    let n = 120;
    let m = 800;
    let mut rng = StdRng::seed_from_u64(17);
    let (tails, heads, lengths) = random_graph(n, m, &mut rng);
    let gradients: Vec<f64> = (0..m).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let (mut chain, _) = BranchingTreeChain::build(n, &tails, &heads, &lengths, None, 0.125, true)
        .expect("chain should build");

    let kappa = (m as f64).powf(-0.01);
    let Some(cycle) = chain.extract_min_ratio_cycle(&gradients, &lengths, kappa) else {
        return;
    };
    let (approx_num, approx_den) = chain.compute_cycle_metrics(&cycle, &gradients, &lengths);
    assert!(approx_den.is_finite());
    assert!(approx_num.is_finite());

    let decomposition = chain.decompose_circulation(&cycle).expect("decomposes");
    let log_n = (n as f64).ln().ceil() as usize;
    assert!(decomposition.tree_paths.len() <= 2 * log_n.max(1));
    assert!(decomposition.off_tree_edges.len() <= (m as f64).sqrt().ceil() as usize);

    let mut recomposed = Vec::new();
    for path in &decomposition.tree_paths {
        recomposed.extend(path);
    }
    recomposed.extend(decomposition.off_tree_edges.iter().copied());
    assert!(!recomposed.is_empty());
    let _ = exact_best_cycle(n, &tails, &heads, &gradients, &lengths);
}
