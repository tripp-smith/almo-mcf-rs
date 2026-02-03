use almo_mcf_core::min_ratio::branching_tree_chain::{exact_best_cycle, BranchingTreeChain};
use almo_mcf_core::min_ratio::{static_oracle::StaticOracle, MinRatioOracle, OracleQuery};

#[test]
fn min_ratio_oracle_detects_negative_cycle() {
    let tails = vec![0, 1, 2, 0];
    let heads = vec![1, 2, 0, 2];
    let gradients = vec![-2.0, -1.0, -3.0, 4.0];
    let lengths = vec![1.0, 1.0, 1.0, 2.0];
    let mut oracle = MinRatioOracle::new(5, 2);
    let best = oracle
        .best_cycle(OracleQuery {
            iter: 0,
            node_count: 3,
            tails: &tails,
            heads: &heads,
            gradients: &gradients,
            lengths: &lengths,
        })
        .unwrap()
        .unwrap();
    assert!(best.ratio < 0.0);
}

#[test]
fn static_oracle_returns_cycle_candidate() {
    let tails = vec![0, 1, 2];
    let heads = vec![1, 2, 0];
    let gradients = vec![1.0, -4.0, 1.5];
    let lengths = vec![1.0, 2.0, 1.0];
    let oracle = StaticOracle::new(19);
    let best = oracle
        .best_cycle(3, &tails, &heads, &gradients, &lengths)
        .unwrap()
        .unwrap();
    assert!(best.denominator > 0.0);
}

#[test]
fn branching_tree_chain_builds_and_extracts_cycles() {
    let tails = vec![0, 1, 2];
    let heads = vec![1, 2, 0];
    let gradients = vec![-1.0, -0.5, -0.25];
    let lengths = vec![1.0, 1.0, 1.0];
    let (mut chain, logs) =
        BranchingTreeChain::build(3, &tails, &heads, &lengths, None, 0.875, true).unwrap();
    assert!(!chain.levels.is_empty());
    assert_eq!(chain.levels[0].tails.len(), tails.len());
    assert_eq!(logs.len(), chain.levels.len());
    let cycle = chain.extract_fundamental_cycle(0, 0).expect("cycle");
    let metrics = chain
        .compute_cycle_metrics(0, &cycle, &gradients, &lengths)
        .expect("metrics");
    assert!(metrics.tilde_length > 0.0);
}

#[test]
fn branching_tree_chain_approximates_best_cycle() {
    let tails = vec![0, 1, 2, 0];
    let heads = vec![1, 2, 0, 2];
    let gradients = vec![-2.0, -1.0, -3.0, 4.0];
    let lengths = vec![1.0, 1.0, 1.0, 2.0];
    let (mut chain, _) =
        BranchingTreeChain::build(3, &tails, &heads, &lengths, None, 0.875, true).unwrap();
    let candidate = chain
        .find_max_ratio_cycle(&gradients, &lengths, 16)
        .expect("candidate");
    let exact = exact_best_cycle(3, &tails, &heads, &gradients, &lengths).expect("exact");
    assert!(candidate.ratio.is_finite());
    assert!(exact.1 <= 0.0);
}

#[test]
fn branching_tree_chain_decomposes_circulation() {
    let tails = vec![0, 1, 2];
    let heads = vec![1, 2, 0];
    let gradients = vec![-1.0, -0.5, -0.25];
    let lengths = vec![1.0, 1.0, 1.0];
    let (mut chain, _) =
        BranchingTreeChain::build(3, &tails, &heads, &lengths, None, 0.875, true).unwrap();
    let cycle = chain.extract_fundamental_cycle(0, 1).expect("cycle");
    let circulation = chain
        .decompose_to_circulation(&cycle, &gradients, &lengths)
        .expect("circulation");
    assert!(circulation.flow_value >= 0.0);
    assert!(!circulation.tree_paths.is_empty());
}
