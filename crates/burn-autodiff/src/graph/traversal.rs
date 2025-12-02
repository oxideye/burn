use super::{Step, StepBoxed};
use crate::{
    NodeId,
    collections::{HashMap, HashSet},
    graph::Parent,
};
use alloc::vec::Vec;

/// Breadth for search algorithm.
pub struct BreadthFirstSearch;

/// Trait for items that can be traversed in the computation graph.
pub trait TraversalItem {
    /// Returns the node ID of this item.
    fn id(&self) -> NodeId;
    /// Returns the parent references of this item.
    fn parents(&self) -> &[Parent];
    /// Returns the node IDs of all parents.
    fn parent_nodes(&self) -> Vec<NodeId> {
        self.parents().iter().map(|p| p.id).collect()
    }
}

impl BreadthFirstSearch {
    /// Traverse the graph of backward steps from a root node.
    pub fn traverse<F, I>(
        &self,
        root_id: NodeId,
        root_step: I,
        steps: &mut HashMap<NodeId, I>,
        mut callback: F,
    ) where
        F: FnMut(NodeId, I),
        I: TraversalItem,
    {
        let mut visited = HashSet::new();
        let mut parents = Vec::new();

        visited.insert(root_id);
        parents.append(&mut root_step.parent_nodes());

        callback(root_id, root_step);

        while let Some(id) = parents.pop() {
            let step = match steps.remove(&id) {
                Some(step) => step,
                None => continue,
            };

            let step_node = step.id();
            let step_parents = step.parent_nodes();

            if visited.contains(&step_node) {
                continue;
            }

            visited.insert(step_node);

            for id in step_parents.iter() {
                if !visited.contains(id) {
                    parents.push(*id);
                }
            }

            callback(step_node, step);
        }
    }
}

impl TraversalItem for StepBoxed {
    fn id(&self) -> NodeId {
        Step::node(self.as_ref())
    }

    fn parents(&self) -> &[Parent] {
        Step::parents(self.as_ref())
    }
}
