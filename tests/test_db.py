import uuid
from datetime import datetime
from autodiscovery.models import Node, BeliefSample, AgentInvocation
from autodiscovery.db import Database


def _make_node(exploration_id="exp-1", parent_id=None, status="pending", **overrides):
    defaults = dict(
        id=str(uuid.uuid4()),
        exploration_id=exploration_id,
        hypothesis="Test hypothesis",
        parent_id=parent_id,
        depth=0 if parent_id is None else 1,
        status=status,
    )
    defaults.update(overrides)
    return Node(**defaults)


def test_initialize_creates_tables(tmp_db):
    rows = tmp_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = {r[0] for r in rows}
    assert "nodes" in table_names
    assert "belief_samples" in table_names
    assert "agent_invocations" in table_names


def test_wal_mode_enabled(tmp_db):
    result = tmp_db.execute("PRAGMA journal_mode").fetchone()
    assert result[0] == "wal"


def test_insert_and_get_node(tmp_db):
    node = _make_node(
        hypothesis="H1: dark matter",
        depth=0,
        visit_count=5,
        surprisal_sum=1.23,
        status="verified",
    )
    tmp_db.insert_node(node)
    fetched = tmp_db.get_node(node.id)
    assert fetched.id == node.id
    assert fetched.hypothesis == "H1: dark matter"
    assert fetched.visit_count == 5
    assert fetched.surprisal_sum == 1.23
    assert fetched.status == "verified"
    assert fetched.exploration_id == "exp-1"


def test_get_children(tmp_db):
    parent = _make_node()
    child1 = _make_node(parent_id=parent.id, depth=1)
    child2 = _make_node(parent_id=parent.id, depth=1)
    unrelated = _make_node()

    for n in [parent, child1, child2, unrelated]:
        tmp_db.insert_node(n)

    children = tmp_db.get_children(parent.id)
    child_ids = {c.id for c in children}
    assert child_ids == {child1.id, child2.id}
    assert unrelated.id not in child_ids


def test_get_children_excludes_pruned(tmp_db):
    parent = _make_node()
    alive = _make_node(parent_id=parent.id, depth=1, status="verified")
    pruned = _make_node(parent_id=parent.id, depth=1, status="pruned")

    for n in [parent, alive, pruned]:
        tmp_db.insert_node(n)

    children = tmp_db.get_children(parent.id, exclude_pruned=True)
    assert len(children) == 1
    assert children[0].id == alive.id


def test_update_node_fields(tmp_db):
    node = _make_node(status="pending", visit_count=0)
    tmp_db.insert_node(node)

    tmp_db.update_node(node.id, status="verified", visit_count=10, surprisal_sum=2.5)

    fetched = tmp_db.get_node(node.id)
    assert fetched.status == "verified"
    assert fetched.visit_count == 10
    assert fetched.surprisal_sum == 2.5
    # Unchanged field remains default
    assert fetched.hypothesis == "Test hypothesis"


def test_get_path_to_root(tmp_db):
    root = _make_node(depth=0)
    middle = _make_node(parent_id=root.id, depth=1)
    leaf = _make_node(parent_id=middle.id, depth=2)

    for n in [root, middle, leaf]:
        tmp_db.insert_node(n)

    path = tmp_db.get_path_to_root(leaf.id)
    assert len(path) == 3
    assert path[0].id == leaf.id
    assert path[1].id == middle.id
    assert path[2].id == root.id


def test_insert_belief_sample(tmp_db):
    node = _make_node()
    tmp_db.insert_node(node)

    sample = BeliefSample(
        node_id=node.id,
        phase="prior",
        sample_index=0,
        believes_hypothesis=True,
        raw_response="Yes, I believe this.",
    )
    tmp_db.insert_belief_sample(sample)

    samples = tmp_db.get_belief_samples(node.id, "prior")
    assert len(samples) == 1
    assert samples[0].believes_hypothesis is True
    assert samples[0].raw_response == "Yes, I believe this."
    assert samples[0].phase == "prior"
    assert samples[0].sample_index == 0
    assert samples[0].id is not None


def test_count_completed_iterations(tmp_db):
    exp_id = "exp-count"
    verified = _make_node(exploration_id=exp_id, status="verified")
    failed = _make_node(exploration_id=exp_id, status="failed")
    pending = _make_node(exploration_id=exp_id, status="pending")
    expanding = _make_node(exploration_id=exp_id, status="expanding")

    for n in [verified, failed, pending, expanding]:
        tmp_db.insert_node(n)

    count = tmp_db.count_completed(exp_id)
    assert count == 2


def test_reset_stale_expanding(tmp_db):
    expanding = _make_node(status="expanding", virtual_loss=3)
    pending = _make_node(status="pending", virtual_loss=0)
    verified = _make_node(status="verified", virtual_loss=1)

    for n in [expanding, pending, verified]:
        tmp_db.insert_node(n)

    tmp_db.reset_stale_expanding()

    fetched_ex = tmp_db.get_node(expanding.id)
    assert fetched_ex.status == "pending"
    assert fetched_ex.virtual_loss == 0

    # Others unchanged
    fetched_pend = tmp_db.get_node(pending.id)
    assert fetched_pend.status == "pending"
    assert fetched_pend.virtual_loss == 0

    fetched_ver = tmp_db.get_node(verified.id)
    assert fetched_ver.status == "verified"
    assert fetched_ver.virtual_loss == 1


def test_node_visit_stats_view(tmp_db):
    parent = _make_node(visit_count=100)
    child = _make_node(
        parent_id=parent.id,
        depth=1,
        visit_count=20,
        surprisal_sum=5.0,
    )

    for n in [parent, child]:
        tmp_db.insert_node(n)

    rows = tmp_db.execute(
        "SELECT node_id, exploit_score, parent_visits FROM node_visit_stats WHERE node_id = ?",
        (child.id,),
    ).fetchall()
    assert len(rows) == 1
    node_id, exploit_score, parent_visits = rows[0]
    assert node_id == child.id
    # exploit_score = surprisal_sum / visit_count = 5.0 / 20 = 0.25
    assert abs(exploit_score - 0.25) < 1e-9
    assert parent_visits == 100
