import json
from surprisal.models import Node
from surprisal.workspace import (
    create_workspace, write_branch_context, write_claude_md,
    assign_branch_id, copy_parent_memory, get_experiment_dir,
)


def test_create_workspace_makes_dirs(tmp_path):
    ws = create_workspace(tmp_path, "branch_abc")
    assert (ws / "memory").is_dir()
    assert (ws / "experiments").is_dir()
    assert (ws / "context").is_dir()


def test_create_workspace_idempotent(tmp_path):
    ws1 = create_workspace(tmp_path, "branch_abc")
    (ws1 / "memory" / "notes.md").write_text("existing")
    ws2 = create_workspace(tmp_path, "branch_abc")
    assert ws1 == ws2
    assert (ws2 / "memory" / "notes.md").read_text() == "existing"


def test_write_branch_context(tmp_path):
    ws = create_workspace(tmp_path, "b1")
    nodes = [
        Node(id="leaf", exploration_id="e", hypothesis="leaf h", depth=2, parent_id="mid"),
        Node(id="mid", exploration_id="e", hypothesis="mid h", depth=1, parent_id="root"),
        Node(id="root", exploration_id="e", hypothesis="root h", depth=0),
    ]
    write_branch_context(ws, nodes)
    history = json.loads((ws / "context" / "branch_history.json").read_text())
    assert len(history) == 3
    assert history[0]["id"] == "root"
    assert history[2]["id"] == "leaf"


def test_write_claude_md(tmp_path):
    ws = create_workspace(tmp_path, "b1")
    nodes = [
        Node(id="leaf", exploration_id="e", hypothesis="leaf h", depth=1, parent_id="root"),
        Node(id="root", exploration_id="e", hypothesis="root h", depth=0),
    ]
    write_claude_md(ws, "AI for science", nodes)
    content = (ws / "CLAUDE.md").read_text()
    assert "AI for science" in content
    assert "root h" in content
    assert "leaf h" in content


def test_assign_branch_id_inherits():
    bid = assign_branch_id("parent_branch", parent_has_other_children=False)
    assert bid == "parent_branch"


def test_assign_branch_id_diverges():
    bid = assign_branch_id("parent_branch", parent_has_other_children=True)
    assert bid != "parent_branch"
    assert len(bid) == 8


def test_copy_parent_memory(tmp_path):
    parent_ws = create_workspace(tmp_path, "parent")
    child_ws = create_workspace(tmp_path, "child")
    (parent_ws / "memory" / "insights.md").write_text("parent insights")
    copy_parent_memory(parent_ws, child_ws)
    assert (child_ws / "memory" / "insights.md").read_text() == "parent insights"


def test_copy_parent_memory_recurses_into_nested_directories(tmp_path):
    parent_ws = create_workspace(tmp_path, "parent")
    child_ws = create_workspace(tmp_path, "child")
    nested_dir = parent_ws / "memory" / "papers" / "2026"
    nested_dir.mkdir(parents=True)
    (nested_dir / "summary.md").write_text("nested memory")

    copy_parent_memory(parent_ws, child_ws)

    assert (child_ws / "memory" / "papers" / "2026" / "summary.md").read_text() == "nested memory"


def test_get_experiment_dir(tmp_path):
    ws = create_workspace(tmp_path, "b1")
    exp_dir = get_experiment_dir(ws, "node123")
    assert exp_dir.is_dir()
    assert exp_dir.name == "node_node123"
