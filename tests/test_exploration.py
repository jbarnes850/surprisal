from surprisal.exploration import (
    create_exploration,
    find_latest_exploration,
    load_branch_sessions,
    load_exploration,
    save_branch_sessions,
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
    import time

    time.sleep(0.01)  # ensure different timestamps
    create_exploration(tmp_path, domain="second")
    latest = find_latest_exploration(tmp_path)
    assert latest is not None
    loaded = load_exploration(latest)
    assert loaded.domain == "second"


def test_find_latest_exploration_empty(tmp_path):
    assert find_latest_exploration(tmp_path) is None


def test_branch_sessions_round_trip(tmp_path):
    exp = create_exploration(tmp_path, domain="session domain", budget=10)
    exp_dir = tmp_path / exp.id
    save_branch_sessions(
        exp_dir,
        "root-a",
        research_claude_session_id="claude-123",
        code_session_id="codex-456",
        code_provider="codex",
        runner_claude_session_id="runner-789",
    )

    sessions = load_branch_sessions(exp_dir, "root-a")

    assert sessions["research_claude_session_id"] == "claude-123"
    assert sessions["claude_session_id"] == "claude-123"
    assert sessions["code_session_id"] == "codex-456"
    assert sessions["code_provider"] == "codex"
    assert sessions["codex_session_id"] == "codex-456"
    assert sessions["runner_claude_session_id"] == "runner-789"
