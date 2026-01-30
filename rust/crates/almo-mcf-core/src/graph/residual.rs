use crate::graph::{EdgeId, Graph, NodeId};
use crate::numerics::EPSILON;
use crate::McfError;

#[derive(Debug, Clone)]
pub struct ResidualEdge {
    pub edge_id: EdgeId,
    pub from: NodeId,
    pub to: NodeId,
    pub residual_capacity: f64,
    pub cost: f64,
    pub direction: i8,
}

#[derive(Debug)]
pub struct ResidualGraph<'a> {
    graph: &'a Graph,
    flow: &'a [f64],
    epsilon: f64,
}

impl<'a> ResidualGraph<'a> {
    pub fn new(graph: &'a Graph, flow: &'a [f64]) -> Result<Self, McfError> {
        if flow.len() != graph.edge_count() {
            return Err(McfError::InvalidInput("flow length mismatch".to_string()));
        }
        Ok(Self {
            graph,
            flow,
            epsilon: EPSILON,
        })
    }

    pub fn with_epsilon(graph: &'a Graph, flow: &'a [f64], epsilon: f64) -> Result<Self, McfError> {
        if flow.len() != graph.edge_count() {
            return Err(McfError::InvalidInput("flow length mismatch".to_string()));
        }
        Ok(Self {
            graph,
            flow,
            epsilon,
        })
    }

    pub fn outgoing(&self, node: NodeId) -> Result<Vec<ResidualEdge>, McfError> {
        let mut edges = Vec::new();
        let outgoing = self.graph.outgoing_edges(node)?;
        for edge_id in outgoing {
            let edge = self
                .graph
                .edge(edge_id)
                .ok_or_else(|| McfError::InvalidInput("edge id out of range".to_string()))?;
            let flow = self.flow[edge_id.0];
            let residual = edge.upper - flow;
            if residual > self.epsilon {
                edges.push(ResidualEdge {
                    edge_id,
                    from: edge.tail,
                    to: edge.head,
                    residual_capacity: residual,
                    cost: edge.cost,
                    direction: 1,
                });
            }
        }

        let incoming = self.graph.incoming_edges(node)?;
        for edge_id in incoming {
            let edge = self
                .graph
                .edge(edge_id)
                .ok_or_else(|| McfError::InvalidInput("edge id out of range".to_string()))?;
            let flow = self.flow[edge_id.0];
            let residual = flow - edge.lower;
            if residual > self.epsilon {
                edges.push(ResidualEdge {
                    edge_id,
                    from: edge.head,
                    to: edge.tail,
                    residual_capacity: residual,
                    cost: -edge.cost,
                    direction: -1,
                });
            }
        }

        Ok(edges)
    }
}
