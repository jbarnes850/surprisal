import json
import pytest
from pathlib import Path
from autodiscovery.exploration import (
    create_exploration, load_exploration, find_latest_exploration,
    update_session, get_session_ids, ExplorationMeta,
)


def test_create_exploration_makes_directory_structure(tmp_path):
    exp = create_exploration(tmp_path, domain="AI for science", budget=100)
    exp_dir = tmp_path / exp.id
    assert (exp_dir / "meta.json").exists()
    assert (exp_dir / "sessions.json").exists()
    assert (exp_dir / "workspaces").is_dir()
    assert (exp_dir / "prompts").is_dir()
    assert (exp_dir / "logs").is_dir()
    assert (exp_dir / "dedup").is_dir()


def test_meta_json_round_trip(tmp_path):
    exp = create_exploration(tmp_path, domain="test domain", budget=50)
    loaded = load_exploration(tmp_path / exp.id)
    assert loaded.domain == "test domain"
    assert loaded.budget == 50
    assert loaded.status == "initialized"


def test_create_exploration_idempotent_by_domain(tmp_path):
    exp1 = create_exploration(tmp_path, domain="same domain", budget=100)
    exp2 = create_exploration(tmp_path, domain="same domain", budget=100)
    assert exp1.id == exp2.id


def test_create_exploration_different_domains(tmp_path):
    exp1 = create_exploration(tmp_path, domain="domain A", budget=100)
    exp2 = create_exploration(tmp_path, domain="domain B", budget=100)
    assert exp1.id != exp2.id


def test_find_latest_exploration(tmp_path):
    create_exploration(tmp_path, domain="first")
    import time; time.sleep(0.01)  # ensure different timestamps
    exp2 = create_exploration(tmp_path, domain="second")
    latest = find_latest_exploration(tmp_path)
    assert latest is not None
    loaded = load_exploration(latest)
    assert loaded.domain == "second"


def test_find_latest_exploration_empty(tmp_path):
    assert find_latest_exploration(tmp_path) is None


def test_update_session(tmp_path):
    exp = create_exploration(tmp_path, domain="test", budget=100)
    exp_dir = tmp_path / exp.id
    update_session(exp_dir, branch_id="branch_a",
                   claude_session_id="claude-123", codex_session_id=None)
    sessions = get_session_ids(exp_dir, "branch_a")
    assert sessions["claude_session_id"] == "claude-123"
    assert sessions["codex_session_id"] is None


def test_get_session_missing_branch(tmp_path):
    exp = create_exploration(tmp_path, domain="test", budget=100)
    exp_dir = tmp_path / exp.id
    sessions = get_session_ids(exp_dir, "nonexistent")
    assert sessions is None


def test_update_session_multiple_branches(tmp_path):
    exp = create_exploration(tmp_path, domain="test", budget=100)
    exp_dir = tmp_path / exp.id
    update_session(exp_dir, "b1", claude_session_id="c1")
    update_session(exp_dir, "b2", claude_session_id="c2")
    assert get_session_ids(exp_dir, "b1")["claude_session_id"] == "c1"
    assert get_session_ids(exp_dir, "b2")["claude_session_id"] == "c2"
