use almo_mcf_core::graph::EdgeId;
use almo_mcf_core::hsfc::{
    HSFCSequence, HSFCStability, HSFCWitness, LazyHSFCOracle, LazyOracle, RebuildingGame,
};
use std::collections::HashSet;

#[test]
fn hsfc_sequence_detects_instability() {
    let mut sequence = HSFCSequence::new(2);
    let prev = sequence.latest().clone();
    let witness = HSFCWitness::new(vec![0.0, 3.0]);
    let dirty = HashSet::new();
    assert_eq!(witness.check_stability(&prev, &dirty), false);
    let status = sequence.append(witness, &dirty);
    assert_eq!(status, HSFCStability::Unstable);
}

#[test]
fn lazy_oracle_matches_recent_updates() {
    let mut oracle = LazyHSFCOracle::new(4);
    oracle.push_witness(HSFCWitness::new(vec![1.0, -0.5]));
    oracle.push_witness(HSFCWitness::new(vec![0.5, 0.5]));
    let value = oracle.lazy_gradient(EdgeId(0), 2.0);
    assert!(value > 2.0);
}

#[test]
fn rebuilding_game_triggers_rebuilds() {
    let mut game = RebuildingGame::new(3, 1, 2);
    assert!(game.advance_round(0));
    let rebuilds = game.handle_loss(0);
    assert!(rebuilds.is_empty());
    let rebuilds = game.handle_loss(0);
    assert_eq!(rebuilds, vec![0]);
}
