use alloc::{sync::Arc, vec::Vec};

#[cfg(target_has_atomic = "64")]
use core::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(target_has_atomic = "64"))]
use portable_atomic::{AtomicU64, Ordering};

use crate::checkpoint::retro_forward::RetroForward;
use crate::runtime::AutodiffClientImpl;

use super::Requirement;

/// Property describing the computational characteristics of an operation.
///
/// Used to determine checkpointing strategy during backward pass.
#[derive(Debug, Clone)]
pub enum ComputingProperty {
    /// The operation is compute-bound (expensive to recompute).
    ComputeBound,
    /// The operation is memory-bound and can be efficiently recomputed.
    MemoryBound {
        /// The function to recompute the forward pass.
        retro_forward: Arc<dyn RetroForward>,
    },
    /// The operation's characteristics are unknown or mixed.
    Ambiguous,
}

/// This is safe only because we only call RetroForward on the autodiff server.
/// Therefore, the trait will never be used by multiple threads at the same time.
///
/// TODO: Find a way to avoid cloning the compute property, which will remove the need to add the
/// Arc, which will make (dyn RetroForward) safely implement Send.
unsafe impl Send for ComputingProperty {}
/// unsafe Sync is required because Send is only implemented for Arc<Sync>, not Arc<Send>.
unsafe impl Sync for ComputingProperty {}

/// A node contains graph metadata and should be used wrapped in an Arc for cheap cloning.
#[derive(new, Debug)]
pub struct Node {
    /// The parent nodes in the computation graph.
    pub parents: Vec<Parent>,
    /// The topological order (depth) of this node in the graph.
    pub order: usize,
    /// The unique identifier for this node.
    pub id: NodeId,
    /// The gradient requirement for this node.
    pub requirement: Requirement,
    /// The computing property determining checkpointing behavior.
    pub properties: ComputingProperty,
    /// The autodiff client for registering operations.
    pub client: AutodiffClientImpl,
}
/// Reference-counted pointer to a Node.
pub type NodeRef = Arc<Node>;

/// A parent reference in the computation graph.
#[derive(new, Debug, Clone, PartialEq, Eq)]
pub struct Parent {
    /// The node ID of the parent.
    pub id: NodeId,
}

impl Node {
    /// Returns the [node](Node) only if gradients are required.
    pub fn clone_if_require_grad(self: &Arc<Self>) -> Option<NodeRef> {
        match self.requirement.is_none() {
            true => None,
            false => Some(self.clone()),
        }
    }
}

/// Unique identifier generated for each node.
#[derive(Clone, Hash, PartialEq, Eq, Debug, Copy)]
pub struct NodeId {
    /// The integer representation of the id
    pub value: u64,
}

impl core::fmt::Display for NodeId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("NodeId({})", self.value))
    }
}

impl NodeId {
    /// Create a unique [node id](NodeId).
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let value = COUNTER.fetch_add(1, Ordering::Relaxed);
        if value == u64::MAX {
            panic!("NodeId overflowed");
        }
        Self { value }
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}
