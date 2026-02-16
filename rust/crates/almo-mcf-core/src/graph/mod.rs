use std::collections::HashMap;

use crate::McfError;

mod core;
pub mod matrix;
pub(crate) mod min_cost_flow;
pub mod residual;
pub mod undirected;

pub use core::{CoreEdge, CoreGraph, Edge, EdgeId, Graph, NodeId};

#[derive(Debug, Clone)]
pub struct IdMapping {
    internal_to_external: Vec<u32>,
    external_to_internal: HashMap<u32, usize>,
}

impl IdMapping {
    pub fn from_range(n: u32) -> Self {
        let internal_to_external: Vec<u32> = (0..n).collect();
        let external_to_internal = internal_to_external
            .iter()
            .enumerate()
            .map(|(idx, &id)| (id, idx))
            .collect();
        Self {
            internal_to_external,
            external_to_internal,
        }
    }

    pub fn from_external_ids(ids: Vec<u32>) -> Result<Self, McfError> {
        let mut external_to_internal = HashMap::with_capacity(ids.len());
        for (idx, &id) in ids.iter().enumerate() {
            if external_to_internal.insert(id, idx).is_some() {
                return Err(McfError::InvalidInput(
                    "duplicate external node id".to_string(),
                ));
            }
        }
        Ok(Self {
            internal_to_external: ids,
            external_to_internal,
        })
    }

    pub fn external_id(&self, internal: usize) -> Option<u32> {
        self.internal_to_external.get(internal).copied()
    }

    pub fn internal_id(&self, external: u32) -> Option<usize> {
        self.external_to_internal.get(&external).copied()
    }

    pub fn len(&self) -> usize {
        self.internal_to_external.len()
    }

    pub fn is_empty(&self) -> bool {
        self.internal_to_external.is_empty()
    }
}
