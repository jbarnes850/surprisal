import json
import shutil
import uuid
from pathlib import Path

from surprisal.models import Node


def create_workspace(base_dir: Path, branch_id: str) -> Path:
    """Create a branch workspace directory with standard structure."""
    ws = base_dir / branch_id
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "memory").mkdir(exist_ok=True)
    (ws / "experiments").mkdir(exist_ok=True)
    (ws / "context").mkdir(exist_ok=True)
    return ws


def write_branch_context(workspace: Path, nodes_path: list[Node]) -> None:
    """Write branch_history.json with full path from root to current node."""
    history = []
    for node in reversed(nodes_path):  # nodes_path is [leaf, ..., root], reverse to [root, ..., leaf]
        entry = {
            "id": node.id,
            "hypothesis": node.hypothesis,
            "context": node.context,
            "variables": node.variables,
            "relationships": node.relationships,
            "depth": node.depth,
            "status": node.status,
        }
        history.append(entry)
    context_file = workspace / "context" / "branch_history.json"
    context_file.write_text(json.dumps(history, indent=2))


def write_claude_md(workspace: Path, domain: str, nodes_path: list[Node]) -> None:
    """Write CLAUDE.md with domain and branch context for Claude agents."""
    lines = [
        "# AutoDiscovery Branch Context",
        "",
        "## Domain",
        f"{domain}",
        "",
        "## Branch History (root → current)",
    ]
    for node in reversed(nodes_path):
        lines.append(f"- **Depth {node.depth}**: {node.hypothesis}")
        if node.context:
            lines.append(f"  Context: {node.context}")
    lines.append("")
    lines.append("## Instructions")
    lines.append("You are exploring this branch of the hypothesis tree. Build on or diverge from the prior findings above.")
    lines.append("Write insights to the memory/ directory. Read it at the start of each session.")
    (workspace / "CLAUDE.md").write_text("\n".join(lines))


def assign_branch_id(parent_branch_id: str, parent_has_other_children: bool) -> str:
    """Assign a branch ID for a new child node.
    If parent has no other children, inherit parent's branch_id.
    If parent already has children, create a new branch (divergence)."""
    if parent_has_other_children:
        return uuid.uuid4().hex[:8]
    return parent_branch_id


def copy_parent_memory(parent_workspace: Path, child_workspace: Path) -> None:
    """Copy parent's memory/ directory to child workspace on branch divergence."""
    parent_mem = parent_workspace / "memory"
    child_mem = child_workspace / "memory"
    if parent_mem.exists():
        shutil.copytree(parent_mem, child_mem, dirs_exist_ok=True)


def get_experiment_dir(workspace: Path, node_id: str) -> Path:
    """Get or create the experiment directory for a specific node."""
    exp_dir = workspace / "experiments" / f"node_{node_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir
