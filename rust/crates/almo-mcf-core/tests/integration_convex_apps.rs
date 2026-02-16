use almo_mcf_core::convex::{entropy_regularized_ot, isotonic_regression, BipartiteGraph, Dag};

#[test]
fn integration_convex_apps() {
    let bg = BipartiteGraph {
        left: 3,
        right: 3,
        edges: vec![(0, 0), (0, 1), (1, 1), (2, 2)],
    };
    let (_f, c) = entropy_regularized_ot(&bg, &[1.0, 1.0, 1.0, -1.0, -1.0, -1.0], &[1.0; 4], 0.1);
    assert!(c.is_finite());

    let dag = Dag {
        n: 4,
        edges: vec![(0, 1), (1, 2), (2, 3)],
    };
    let x = isotonic_regression(&dag, &[4.0, 3.0, 2.0, 1.0], 2.0);
    for (u, v) in dag.edges {
        assert!(x[u] <= x[v] + 1e-8);
    }
}
