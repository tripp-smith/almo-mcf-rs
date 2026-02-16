use almo_mcf_core::rebuilding::RebuildingGame;

#[test]
fn test_derandomized_reproducibility() {
    let mut a = RebuildingGame::new(400, 2000, 5);
    let mut b = RebuildingGame::new(400, 2000, 5);
    a.enable_derandomization(9);
    b.enable_derandomization(9);
    let mut da = Vec::new();
    let mut db = Vec::new();
    for t in 1..200 {
        da.push(a.play_round(t, 2, t % 3 != 0).level);
        db.push(b.play_round(t, 2, t % 3 != 0).level);
    }
    assert_eq!(da, db);
    assert!(a.verify_derandomized_invariants());
}
