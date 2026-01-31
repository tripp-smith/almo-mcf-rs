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
