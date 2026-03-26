import pytest
import asyncio

from surprisal.config import AutoDiscoveryConfig
from surprisal.db import Database
from surprisal.models import Node
from surprisal.orchestrator import AtomicCounter, DedupCheckpoint, run_exploration, worker_loop
from surprisal.providers import LiteratureStatus, ProviderStatus
from surprisal.workspace import create_workspace


@pytest.mark.asyncio
async def test_atomic_counter_decrements():
    counter = AtomicCounter(3)
    assert await counter.decrement() is True
    assert await counter.decrement() is True
    assert await counter.decrement() is True
    assert await counter.decrement() is False


@pytest.mark.asyncio
async def test_atomic_counter_zero():
    counter = AtomicCounter(0)
    assert await counter.decrement() is False


@pytest.mark.asyncio
async def test_run_exploration_scopes_budget_to_current_exploration(tmp_path, monkeypatch):
    db = Database(tmp_path / "tree.db")
    db.initialize()
    root = Node(
        id="root-a",
        exploration_id="exp-a",
        hypothesis="Root A",
        status="verified",
        visit_count=1,
        branch_id="root",
    )
    other_verified = Node(
        id="root-b",
        exploration_id="exp-b",
        hypothesis="Root B",
        status="verified",
        visit_count=1,
        branch_id="root-b",
    )
    other_failed = Node(
        id="child-b",
        exploration_id="exp-b",
        hypothesis="Child B",
        parent_id="root-b",
        depth=1,
        status="failed",
    )
    for node in [root, other_verified, other_failed]:
        db.insert_node(node)

    async def fake_worker_loop(*args, **kwargs):
        return None

    monkeypatch.setattr("surprisal.orchestrator.worker_loop", fake_worker_loop)

    result = await run_exploration(
        db=db,
        exploration_dir=tmp_path,
        budget=1,
        concurrency=1,
        c_explore=1.414,
        config=AutoDiscoveryConfig(),
        root_id="root-a",
        domain="test",
        providers=ProviderStatus(claude_available=True, codex_available=False),
        literature_provider=LiteratureStatus(provider="huggingface"),
    )

    assert result["status"] == "completed"
    assert result["iterations"] == 0
    db.close()


@pytest.mark.asyncio
async def test_worker_loop_copies_memory_and_runs_dedup(tmp_path, monkeypatch):
    db = Database(tmp_path / "tree.db")
    db.initialize()
    root = Node(
        id="root",
        exploration_id="exp-a",
        hypothesis="Root",
        status="verified",
        visit_count=4,
        branch_id="root",
    )
    existing_child = Node(
        id="existing",
        exploration_id="exp-a",
        hypothesis="Existing",
        parent_id="root",
        depth=1,
        status="verified",
        visit_count=1,
        branch_id="root",
    )
    for node in [root, existing_child]:
        db.insert_node(node)

    workspaces_dir = tmp_path / "workspaces"
    root_ws = create_workspace(workspaces_dir, "root")
    (root_ws / "memory" / "notes.md").write_text("carry me forward")

    async def fake_run_live_fsm(node_id, db, **kwargs):
        db.update_node(node_id, status="verified", virtual_loss=0, belief_shifted=False)
        return True

    dedup_calls = []

    def fake_deduplicate(db, max_distance=0.5, exploration_id=None):
        dedup_calls.append(exploration_id)
        return {"clusters": 0, "duplicates": 0, "total_nodes": 2}

    monkeypatch.setattr("surprisal.fsm_runner.run_live_fsm", fake_run_live_fsm)
    monkeypatch.setattr("surprisal.orchestrator.deduplicate", fake_deduplicate)

    cfg = AutoDiscoveryConfig()
    cfg.mcts.dedup_interval = 1
    counter = AtomicCounter(1)
    await worker_loop(
        db=db,
        exploration_dir=tmp_path,
        selection_lock=asyncio.Lock(),
        counter=counter,
        dedup_lock=asyncio.Lock(),
        dedup_checkpoint=DedupCheckpoint(),
        shutdown=asyncio.Event(),
        c_explore=1.414,
        config=cfg,
        root_id="root",
        domain="test",
        providers=ProviderStatus(claude_available=True, codex_available=False),
        literature_provider=LiteratureStatus(provider="huggingface"),
    )

    child_workspaces = [p for p in workspaces_dir.iterdir() if p.name != "root"]
    assert len(child_workspaces) == 1
    assert (child_workspaces[0] / "memory" / "notes.md").read_text() == "carry me forward"
    assert dedup_calls == ["exp-a"]
    db.close()


@pytest.mark.asyncio
async def test_run_exploration_surfaces_worker_failures(tmp_path, monkeypatch):
    db = Database(tmp_path / "tree.db")
    db.initialize()
    root = Node(
        id="root",
        exploration_id="exp-a",
        hypothesis="Root",
        status="verified",
        visit_count=1,
        branch_id="root",
    )
    db.insert_node(root)

    async def fake_run_live_fsm(node_id, db, **kwargs):
        db.update_node(node_id, status="verified", virtual_loss=0, belief_shifted=False)
        return True

    def fake_deduplicate(db, max_distance=0.5, exploration_id=None):
        raise RuntimeError(f"dedup failed for {exploration_id}")

    monkeypatch.setattr("surprisal.fsm_runner.run_live_fsm", fake_run_live_fsm)
    monkeypatch.setattr("surprisal.orchestrator.deduplicate", fake_deduplicate)

    cfg = AutoDiscoveryConfig()
    cfg.mcts.dedup_interval = 1
    result = await run_exploration(
        db=db,
        exploration_dir=tmp_path,
        budget=1,
        concurrency=1,
        c_explore=1.414,
        config=cfg,
        root_id="root",
        domain="test",
        providers=ProviderStatus(claude_available=True, codex_available=False),
        literature_provider=LiteratureStatus(provider="huggingface"),
    )

    assert result["status"] == "error"
    assert "RuntimeError: dedup failed for exp-a" in result["message"]
    assert result["worker_errors"] == ["RuntimeError: dedup failed for exp-a"]
    db.close()


@pytest.mark.asyncio
async def test_worker_loop_dedups_once_per_completed_boundary(tmp_path, monkeypatch):
    db = Database(tmp_path / "tree.db")
    db.initialize()
    root = Node(
        id="root",
        exploration_id="exp-a",
        hypothesis="Root",
        status="verified",
        visit_count=4,
        branch_id="root",
    )
    existing_child = Node(
        id="existing",
        exploration_id="exp-a",
        hypothesis="Existing",
        parent_id="root",
        depth=1,
        status="verified",
        visit_count=1,
        branch_id="root",
    )
    for node in [root, existing_child]:
        db.insert_node(node)

    async def fake_run_live_fsm(node_id, db, **kwargs):
        db.update_node(node_id, status="verified", virtual_loss=0, belief_shifted=False)
        return True

    dedup_calls = []

    def fake_deduplicate(db, max_distance=0.5, exploration_id=None):
        dedup_calls.append(exploration_id)
        return {"clusters": 0, "duplicates": 0, "total_nodes": 2}

    monkeypatch.setattr("surprisal.fsm_runner.run_live_fsm", fake_run_live_fsm)
    monkeypatch.setattr("surprisal.orchestrator.deduplicate", fake_deduplicate)

    cfg = AutoDiscoveryConfig()
    cfg.mcts.dedup_interval = 1
    checkpoint = DedupCheckpoint()
    counter = AtomicCounter(1)

    await worker_loop(
        db=db,
        exploration_dir=tmp_path,
        selection_lock=asyncio.Lock(),
        counter=counter,
        dedup_lock=asyncio.Lock(),
        dedup_checkpoint=checkpoint,
        shutdown=asyncio.Event(),
        c_explore=1.414,
        config=cfg,
        root_id="root",
        domain="test",
        providers=ProviderStatus(claude_available=True, codex_available=False),
        literature_provider=LiteratureStatus(provider="huggingface"),
    )
    counter = AtomicCounter(0)
    await worker_loop(
        db=db,
        exploration_dir=tmp_path,
        selection_lock=asyncio.Lock(),
        counter=counter,
        dedup_lock=asyncio.Lock(),
        dedup_checkpoint=checkpoint,
        shutdown=asyncio.Event(),
        c_explore=1.414,
        config=cfg,
        root_id="root",
        domain="test",
        providers=ProviderStatus(claude_available=True, codex_available=False),
        literature_provider=LiteratureStatus(provider="huggingface"),
    )

    assert dedup_calls == ["exp-a"]
    assert checkpoint.last_completed == 2
    db.close()
