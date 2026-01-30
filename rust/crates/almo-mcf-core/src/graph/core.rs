use crate::numerics::EPSILON;
use crate::McfError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(pub usize);

#[derive(Debug, Clone)]
pub struct Edge {
    pub tail: NodeId,
    pub head: NodeId,
    pub lower: f64,
    pub upper: f64,
    pub cost: f64,
    pub rev: Option<EdgeId>,
}

#[derive(Debug, Clone, Copy)]
pub struct EdgeSpec {
    pub lower: f64,
    pub upper: f64,
    pub cost: f64,
}

#[derive(Debug, Clone)]
pub struct Graph {
    demands: Vec<f64>,
    edges: Vec<Edge>,
    outgoing: Vec<Vec<EdgeId>>,
    incoming: Vec<Vec<EdgeId>>,
}

impl Graph {
    pub fn new(node_count: usize) -> Self {
        Self {
            demands: vec![0.0; node_count],
            edges: Vec::new(),
            outgoing: vec![Vec::new(); node_count],
            incoming: vec![Vec::new(); node_count],
        }
    }

    pub fn with_demands(demands: Vec<f64>) -> Self {
        let node_count = demands.len();
        Self {
            demands,
            edges: Vec::new(),
            outgoing: vec![Vec::new(); node_count],
            incoming: vec![Vec::new(); node_count],
        }
    }

    pub fn node_count(&self) -> usize {
        self.demands.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn demand(&self, node: NodeId) -> Option<f64> {
        self.demands.get(node.0).copied()
    }

    pub fn set_demand(&mut self, node: NodeId, demand: f64) -> Result<(), McfError> {
        if node.0 >= self.demands.len() {
            return Err(McfError::InvalidInput("node id out of range".to_string()));
        }
        self.demands[node.0] = demand;
        Ok(())
    }

    pub fn add_node(&mut self, demand: f64) -> NodeId {
        let node_id = NodeId(self.demands.len());
        self.demands.push(demand);
        self.outgoing.push(Vec::new());
        self.incoming.push(Vec::new());
        node_id
    }

    pub fn add_edge(
        &mut self,
        tail: NodeId,
        head: NodeId,
        lower: f64,
        upper: f64,
        cost: f64,
    ) -> Result<EdgeId, McfError> {
        if tail.0 >= self.node_count() || head.0 >= self.node_count() {
            return Err(McfError::InvalidInput(
                "edge endpoint outside node range".to_string(),
            ));
        }
        if lower > upper {
            return Err(McfError::InvalidInput(
                "lower bound exceeds upper bound".to_string(),
            ));
        }
        let edge_id = EdgeId(self.edges.len());
        self.edges.push(Edge {
            tail,
            head,
            lower,
            upper,
            cost,
            rev: None,
        });
        self.outgoing[tail.0].push(edge_id);
        self.incoming[head.0].push(edge_id);
        Ok(edge_id)
    }

    pub fn add_edge_pair(
        &mut self,
        tail: NodeId,
        head: NodeId,
        forward: EdgeSpec,
        reverse: EdgeSpec,
    ) -> Result<(EdgeId, EdgeId), McfError> {
        let forward_id = self.add_edge(tail, head, forward.lower, forward.upper, forward.cost)?;
        let reverse_id = self.add_edge(head, tail, reverse.lower, reverse.upper, reverse.cost)?;
        self.link_reverse(forward_id, reverse_id)?;
        self.link_reverse(reverse_id, forward_id)?;
        Ok((forward_id, reverse_id))
    }

    pub fn link_reverse(&mut self, edge: EdgeId, reverse: EdgeId) -> Result<(), McfError> {
        if edge.0 >= self.edges.len() || reverse.0 >= self.edges.len() {
            return Err(McfError::InvalidInput(
                "reverse edge id out of range".to_string(),
            ));
        }
        self.edges[edge.0].rev = Some(reverse);
        Ok(())
    }

    pub fn edge(&self, edge: EdgeId) -> Option<&Edge> {
        self.edges.get(edge.0)
    }

    pub fn edges(&self) -> impl Iterator<Item = (EdgeId, &Edge)> {
        self.edges
            .iter()
            .enumerate()
            .map(|(idx, edge)| (EdgeId(idx), edge))
    }

    pub fn outgoing_edges(
        &self,
        node: NodeId,
    ) -> Result<impl Iterator<Item = EdgeId> + '_, McfError> {
        self.outgoing
            .get(node.0)
            .map(|edges| edges.iter().copied())
            .ok_or_else(|| McfError::InvalidInput("node id out of range".to_string()))
    }

    pub fn incoming_edges(
        &self,
        node: NodeId,
    ) -> Result<impl Iterator<Item = EdgeId> + '_, McfError> {
        self.incoming
            .get(node.0)
            .map(|edges| edges.iter().copied())
            .ok_or_else(|| McfError::InvalidInput("node id out of range".to_string()))
    }

    pub fn check_feasible(&self) -> Result<(), McfError> {
        let total_demand: f64 = self.demands.iter().sum();
        if total_demand.abs() > EPSILON {
            return Err(McfError::InvalidInput(
                "demands must sum to zero".to_string(),
            ));
        }
        if self.has_negative_cost_cycle() {
            return Err(McfError::InvalidInput(
                "negative cost cycle detected".to_string(),
            ));
        }
        Ok(())
    }

    fn has_negative_cost_cycle(&self) -> bool {
        let n = self.node_count();
        if n == 0 {
            return false;
        }
        let mut dist = vec![0.0_f64; n];
        for _ in 0..n {
            let mut updated = false;
            for edge in &self.edges {
                let u = edge.tail.0;
                let v = edge.head.0;
                let nd = dist[u] + edge.cost;
                if nd < dist[v] - EPSILON {
                    dist[v] = nd;
                    updated = true;
                }
            }
            if !updated {
                return false;
            }
        }
        true
    }
}
