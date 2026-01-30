use crate::graph::{EdgeId, Graph, NodeId};
use crate::McfError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UndirectedEdgeRef {
    pub edge_id: EdgeId,
    pub from: NodeId,
    pub to: NodeId,
    pub direction: i8,
}

#[derive(Debug)]
pub struct UndirectedView<'a> {
    graph: &'a Graph,
}

impl<'a> UndirectedView<'a> {
    pub fn new(graph: &'a Graph) -> Self {
        Self { graph }
    }

    pub fn neighbors(&self, node: NodeId) -> Result<Vec<UndirectedEdgeRef>, McfError> {
        let mut edges = Vec::new();
        let outgoing = self.graph.outgoing_edges(node)?;
        for edge_id in outgoing {
            let edge = self
                .graph
                .edge(edge_id)
                .ok_or_else(|| McfError::InvalidInput("edge id out of range".to_string()))?;
            edges.push(UndirectedEdgeRef {
                edge_id,
                from: edge.tail,
                to: edge.head,
                direction: 1,
            });
        }

        let incoming = self.graph.incoming_edges(node)?;
        for edge_id in incoming {
            let edge = self
                .graph
                .edge(edge_id)
                .ok_or_else(|| McfError::InvalidInput("edge id out of range".to_string()))?;
            if edge.tail == edge.head {
                continue;
            }
            edges.push(UndirectedEdgeRef {
                edge_id,
                from: edge.head,
                to: edge.tail,
                direction: -1,
            });
        }

        Ok(edges)
    }
}
