use almo_mcf_core::convex::{
    diffusion_flow, entropy_regularized_ot, isotonic_regression, matrix_scaling, BipartiteGraph,
    ConvexMcfSolver, Dag, McfSolver, PNorm,
};
use almo_mcf_core::graph::{split_vertices_for_caps, Graph, NodeId};
use almo_mcf_core::{detect_negative_cycle, single_source_shortest_paths, McfOptions, McfProblem};

#[test]
fn test_convex_one_step() {
    let problem = McfProblem::new(vec![0], vec![1], vec![0], vec![10], vec![1], vec![-1, 1])
        .expect("problem");
    let base = McfSolver {
        problem,
        opts: McfOptions::default(),
    };
    let funcs: Vec<Box<dyn almo_mcf_core::convex::ConvexFunc>> = vec![Box::new(PNorm {
        p: 2.0,
        weights: vec![1.0],
    })];
    let mut solver = ConvexMcfSolver::new(base, funcs, 1e-3);
    let x = vec![2.0_f64];
    let grad = solver.one_step_convex_analysis(&x);
    assert!(grad[0].is_finite());
    let old = solver.total_cost(&x);
    let new = solver.total_cost(&vec![x[0] + 0.05 * grad[0]]);
    assert!(new <= old + 1e-6);
}

#[test]
fn test_pnorm_and_ot() {
    let mut g = Graph::new(2);
    g.add_edge(NodeId(0), NodeId(1), 0.0, 5.0, 1.0).unwrap();
    let f = almo_mcf_core::convex::p_norm_min_flow(&g, &[1.0, -1.0], &[1.0], 2.0);
    assert_eq!(f.len(), 1);

    let bg = BipartiteGraph {
        left: 2,
        right: 2,
        edges: vec![(0, 0), (0, 1), (1, 0), (1, 1)],
    };
    let (flow, cost) = entropy_regularized_ot(&bg, &[1.0, 1.0, -1.0, -1.0], &[1.0; 4], 0.5);
    assert_eq!(flow.len(), 4);
    assert!(cost.is_finite());

    let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let (x, _y) = matrix_scaling(&a, &[1.0, 1.0]);
    assert_eq!(x.len(), 2);
}

#[test]
fn test_isotonic_dag() {
    let dag = Dag {
        n: 3,
        edges: vec![(0, 1), (1, 2)],
    };
    let x = isotonic_regression(&dag, &[3.0, 2.0, 1.0], 2.0);
    assert!(x[0] <= x[1] + 1e-9);
    assert!(x[1] <= x[2] + 1e-9);
}

#[test]
fn test_vertex_caps_diffusion() {
    let mut g = Graph::new(3);
    g.add_edge(NodeId(0), NodeId(1), 0.0, 10.0, 1.0).unwrap();
    g.add_edge(NodeId(1), NodeId(2), 0.0, 10.0, 1.0).unwrap();
    let split = split_vertices_for_caps(&g, &[1.0, 2.0, 3.0]);
    assert_eq!(split.node_count(), 6);
    let df = diffusion_flow(&g, &[1.0, -1.0, 0.0], &[1.0, 2.0]);
    assert_eq!(df.len(), 2);
}

#[test]
fn test_neg_cycles_sssp() {
    let mut g = Graph::new(3);
    g.add_edge(NodeId(0), NodeId(1), 0.0, 1.0, 1.0).unwrap();
    g.add_edge(NodeId(1), NodeId(2), 0.0, 1.0, 1.0).unwrap();
    g.add_edge(NodeId(2), NodeId(0), 0.0, 1.0, -4.0).unwrap();

    let cycle = detect_negative_cycle(&g, &[0.0, 0.0, 0.0]);
    assert!(cycle.is_some());

    let mut g2 = Graph::new(3);
    g2.add_edge(NodeId(0), NodeId(1), 0.0, 1.0, 2.0).unwrap();
    g2.add_edge(NodeId(1), NodeId(2), 0.0, 1.0, 3.0).unwrap();
    let d = single_source_shortest_paths(&g2, NodeId(0), &[2.0, 3.0]);
    assert!((d[2] - 5.0).abs() < 1e-6);
}
