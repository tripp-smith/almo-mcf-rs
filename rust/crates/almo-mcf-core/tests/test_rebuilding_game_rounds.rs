use almo_mcf_core::rebuilding::RebuildingGame;

#[test]
fn test_rebuilding_game_rounds() {
    let n = 512usize;
    let m = 2048usize;
    let mut game = RebuildingGame::new(n, m, 6);
    let mut rebuilds = 0usize;
    for t in 1..=1000 {
        let d = game.play_round(t, 3, t % 5 != 0);
        if d.trigger_rebuild {
            rebuilds = rebuilds.saturating_add(1);
        }
    }
    assert!(rebuilds <= ((n as f64).ln().ceil() as usize).saturating_mul(8));
    assert!(game.total_recourse.is_finite());
}
