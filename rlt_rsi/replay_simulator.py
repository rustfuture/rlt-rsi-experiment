import json
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Node:
    """Represents a state in the discovery history (e.g. prompt, intermediate code)."""
    def __init__(self, node_id: str, state: str, reward: float = 0.0, is_terminal: bool = False):
        self.node_id = node_id
        self.state = state
        self.reward = reward
        self.is_terminal = is_terminal
        self.children: List['Edge'] = []

class Edge:
    """Represents an action (e.g. model response) transitioning to a new state."""
    def __init__(self, action: str, target_node: Node):
        self.action = action
        self.target_node = target_node

class ReplaySimulator:
    """
    A replay simulator for Recursive Self-Improvement (RSI) exploration.
    Inspired by 'Dream-RSI' (Zheng et al., 2026).
    
    This simulator loads a history of online exploration traces (a tree of past 
    model decisions and their evaluated outcomes) and allows evaluating different 
    meta-exploration policies (e.g., breadth-first, depth-first, early stopping) 
    entirely offline, without invoking the underlying LLM.
    """
    
    def __init__(self, root: Node):
        self.root = root

    @classmethod
    def from_json_trace(cls, filepath: str) -> 'ReplaySimulator':
        """
        Loads an offline trace. For demonstration, this builds a simple in-memory tree.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Simple parser for a trace format
        nodes = {}
        for n_data in data.get("nodes", []):
            nodes[n_data["id"]] = Node(
                node_id=n_data["id"],
                state=n_data.get("state", ""),
                reward=n_data.get("reward", 0.0),
                is_terminal=n_data.get("is_terminal", False)
            )
            
        for e_data in data.get("edges", []):
            source = nodes[e_data["source"]]
            target = nodes[e_data["target"]]
            source.children.append(Edge(action=e_data.get("action", ""), target_node=target))
            
        root_id = data.get("root_id")
        if not root_id or root_id not in nodes:
            raise ValueError("Invalid root_id in trace.")
            
        return cls(root=nodes[root_id])

    def simulate_policy(self, policy_fn, max_steps: int = 100) -> Dict[str, Any]:
        """
        Runs a meta-exploration policy over the recorded simulator pool.
        The policy_fn decides which branches to explore.
        """
        logger.info(f"Starting dreaming simulation using policy: {policy_fn.__name__}")
        best_reward = 0.0
        steps = 0
        frontier = [self.root]
        
        while frontier and steps < max_steps:
            # The policy decides which node to expand and how many branches to take
            current_node, remaining_frontier = policy_fn(frontier)
            frontier = remaining_frontier
            
            if current_node.reward > best_reward:
                best_reward = current_node.reward
                
            if current_node.is_terminal:
                continue
                
            for edge in current_node.children:
                frontier.append(edge.target_node)
                
            steps += 1
            
        logger.info(f"Simulation ended. Steps: {steps}, Best Reward: {best_reward}")
        return {"steps": steps, "best_reward": best_reward}


# --- Example Policies ---

def depth_first_policy(frontier: List[Node]):
    """Explores the deepest node first (LIFO)."""
    current = frontier.pop(-1)
    return current, frontier

def breadth_first_policy(frontier: List[Node]):
    """Explores level by level (FIFO)."""
    current = frontier.pop(0)
    return current, frontier

def greedy_reward_policy(frontier: List[Node]):
    """Explores the node with the highest current reward."""
    frontier.sort(key=lambda n: n.reward)
    current = frontier.pop(-1)
    return current, frontier

