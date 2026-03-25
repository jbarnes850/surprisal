"""MCTS engine — UCT, progressive widening, selection, backpropagation.

Pure logic over the Database layer; never calls agents.
"""

import math

from autodiscovery.db import Database


def compute_uct(
    visit_count: int,
    virtual_loss: int,
    surprisal_sum: float,
    parent_visits: int,
    c_explore: float,
) -> float:
    """Upper Confidence Bound for Trees.

    UCT(H) = exploit(H) + C * explore(H)
      exploit = surprisal_sum / effective_visits
      explore = sqrt(2 * ln(parent_visits) / effective_visits)

    Returns +inf for unvisited nodes.
    """
    effective = visit_count + virtual_loss
    if effective == 0:
        return float("inf")
    exploit = surprisal_sum / effective
    explore = math.sqrt(2 * math.log(parent_visits) / effective) if parent_visits > 0 else 0.0
    return exploit + c_explore * explore


def max_children(visits: int, k: float = 1.0, alpha: float = 0.5) -> int:
    """Progressive widening: floor(k * visits^alpha).

    Controls how many children a node can have based on visit count.
    """
    if visits <= 0:
        return 0
    return int(k * (visits ** alpha))


def select_node(
    db: Database,
    root_id: str,
    c_explore: float,
    k: float = 1.0,
    alpha: float = 0.5,
    max_depth: int = 30,
) -> str:
    """Traverse tree using UCT + progressive widening to find node to expand.

    Returns node_id of the selected parent for expansion.
    Progressive widening uses visit_count ONLY (not virtual_loss).
    Virtual loss only affects UCT computation.
    """
    current_id = root_id
    while True:
        current = db.get_node(current_id)
        # Depth limit check
        if current.depth >= max_depth:
            return current_id
        children = db.get_children(current_id, exclude_pruned=True)
        threshold = max_children(current.visit_count, k, alpha)
        if len(children) < threshold:
            # Progressive widening allows a new child
            return current_id
        if not children:
            # Leaf node with no room for children (visits=0)
            return current_id
        # Select child with highest UCT
        best_id = None
        best_score = -1.0
        for child in children:
            score = compute_uct(
                child.visit_count,
                child.virtual_loss,
                child.surprisal_sum,
                current.visit_count,
                c_explore,
            )
            if score > best_score:
                best_score = score
                best_id = child.id
        current_id = best_id


def backpropagate(db: Database, node_id: str, surprisal_value: int) -> None:
    """Walk from node to root, incrementing visit_count and surprisal_sum."""
    path = db.get_path_to_root(node_id)
    for node in path:
        db.update_node(
            node.id,
            visit_count=node.visit_count + 1,
            surprisal_sum=node.surprisal_sum + surprisal_value,
        )
