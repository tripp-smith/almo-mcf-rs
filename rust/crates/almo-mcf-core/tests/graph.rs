use almo_mcf_core::graph::matrix::incidence_matrix;
use almo_mcf_core::graph::residual::ResidualGraph;
use almo_mcf_core::graph::undirected::UndirectedView;
use almo_mcf_core::graph::{Graph, NodeId};
use almo_mcf_core::McfError;

#[test]
fn graph_iterators_and_residual_view() {
    let mut graph = Graph::new(3);
    let e0 = graph
        .add_edge(NodeId(0), NodeId(1), 0.0, 10.0, 2.0)
        .unwrap();
    let e1 = graph.add_edge(NodeId(1), NodeId(2), 1.0, 5.0, 1.0).unwrap();

    let outgoing_0: Vec<_> = graph.outgoing_edges(NodeId(0)).unwrap().collect();
    let incoming_2: Vec<_> = graph.incoming_edges(NodeId(2)).unwrap().collect();
    assert_eq!(outgoing_0, vec![e0]);
    assert_eq!(incoming_2, vec![e1]);

    let flow = vec![3.0, 2.0];
    let residual = ResidualGraph::new(&graph, &flow).unwrap();
    let residual_from_0 = residual.outgoing(NodeId(0)).unwrap();
    assert_eq!(residual_from_0.len(), 1);
    assert!(residual_from_0[0].residual_capacity > 6.9);
    let residual_from_1 = residual.outgoing(NodeId(1)).unwrap();
    assert_eq!(residual_from_1.len(), 2);
}

#[test]
fn undirected_view_exposes_both_directions() {
    let mut graph = Graph::new(2);
    graph.add_edge(NodeId(0), NodeId(1), 0.0, 3.0, 1.0).unwrap();
    let view = UndirectedView::new(&graph);
    let neighbors = view.neighbors(NodeId(1)).unwrap();
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].from, NodeId(1));
    assert_eq!(neighbors[0].to, NodeId(0));
    assert_eq!(neighbors[0].direction, -1);
}

#[test]
fn incidence_matrix_matches_edge_endpoints() {
    let mut graph = Graph::new(2);
    graph.add_edge(NodeId(0), NodeId(1), 0.0, 5.0, 1.0).unwrap();
    let matrix = incidence_matrix(&graph);
    assert_eq!(matrix.rows, 2);
    assert_eq!(matrix.cols, 1);
    assert_eq!(matrix.values.len(), 2);
    assert_eq!(matrix.row_indices, vec![0, 1]);
    assert_eq!(matrix.col_indices, vec![0, 0]);
    assert_eq!(matrix.values, vec![-1.0, 1.0]);
}

#[test]
fn feasibility_checks_demands_and_negative_cycles() {
    let mut graph = Graph::with_demands(vec![1.0, -1.0]);
    graph.add_edge(NodeId(0), NodeId(1), 0.0, 5.0, 2.0).unwrap();
    assert!(graph.check_feasible().is_ok());

    graph.set_demand(NodeId(0), 2.0).unwrap();
    let err = graph.check_feasible().unwrap_err();
    assert!(matches!(err, McfError::InvalidInput(_)));

    let mut cycle_graph = Graph::new(2);
    cycle_graph
        .add_edge(NodeId(0), NodeId(1), 0.0, 2.0, -2.0)
        .unwrap();
    cycle_graph
        .add_edge(NodeId(1), NodeId(0), 0.0, 2.0, -2.0)
        .unwrap();
    let err = cycle_graph.check_feasible().unwrap_err();
    assert!(matches!(err, McfError::InvalidInput(_)));
}
