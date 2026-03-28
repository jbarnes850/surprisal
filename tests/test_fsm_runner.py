import json

import pytest

from surprisal.agents.base import AgentResult
from surprisal.config import AutoDiscoveryConfig
from surprisal.db import Database
from surprisal.exploration import load_branch_sessions, save_branch_sessions
from surprisal.fsm_runner import _extract_json, _extract_stage_json, run_live_fsm
from surprisal.models import Node
from surprisal.providers import ProviderStatus


class FakeAgent:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def invoke(self, **kwargs):
        self.calls.append(kwargs.copy())
        assert self._responses, "No fake agent responses remaining"
        return self._responses.pop(0)


class FakeBackend:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def execute(
        self,
        experiment_prompt,
        workspace,
        config,
        system_prompt_file=None,
        session_id=None,
        progress_callback=None,
        **kwargs,
    ):
        self.calls.append({
            "experiment_prompt": experiment_prompt,
            "workspace": workspace,
            "system_prompt_file": system_prompt_file,
            "session_id": session_id,
            "progress_callback": progress_callback,
        })
        assert self._responses, "No fake backend responses remaining"
        response = self._responses.pop(0)
        if callable(response):
            return response(workspace)
        return response


def _json_result(payload: dict, exit_code: int = 0, session_id: str | None = None) -> AgentResult:
    body = dict(payload)
    if session_id is not None and "session_id" not in body:
        body["session_id"] = session_id
    return AgentResult.from_raw(json.dumps(body), exit_code=exit_code, session_id=session_id)


def _provider_error(message: str) -> AgentResult:
    return _json_result({"is_error": True, "result": message})


def _backend_success(workspace, stdout="metric = 0.9", code="print('metric = 0.9')"):
    (workspace / "results.json").write_text(json.dumps({
        "code": code,
        "stdout": stdout,
        "metrics": {"metric": 0.9},
        "error": False,
    }))
    return AgentResult(raw=stdout, exit_code=0)


def _make_db_with_node(tmp_path):
    db = Database(tmp_path / "tree.db")
    db.initialize()
    node = Node(
        id="node-1",
        exploration_id="exp-1",
        hypothesis="Seed hypothesis",
        status="pending",
        branch_id="root",
    )
    db.insert_node(node)
    return db, node


def _patch_runtime(monkeypatch, research_responses, code_responses, backend_responses):
    research_agent = FakeAgent(research_responses)
    code_agent = FakeAgent(code_responses)
    monkeypatch.setattr("surprisal.fsm_runner.ClaudeAgent", lambda *args, **kwargs: research_agent)
    monkeypatch.setattr("surprisal.fsm_runner.CodexAgent", lambda *args, **kwargs: code_agent)
    backend = FakeBackend(backend_responses)
    monkeypatch.setattr("surprisal.fsm_runner.create_backend", lambda *args, **kwargs: backend)

    async def _fake_detect_gpu():
        return False

    monkeypatch.setattr("surprisal.fsm_runner.detect_gpu", _fake_detect_gpu)
    backend.research_agent = research_agent
    backend.code_agent = code_agent
    return backend


def test_extract_stage_json_rejects_provider_error_wrapper():
    data, error = _extract_stage_json(_provider_error("rate limited"), "experiment_generator")
    assert data is None
    assert "rate limited" in error


def test_extract_json_unwraps_successful_claude_envelope():
    result = _json_result({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "session_id": "sess-123",
        "result": json.dumps({
            "hypothesis": "Wrapped hypothesis",
            "experiment_plan": "Use a real dataset and report a metric.",
        }),
    })

    data = _extract_json(result)

    assert data == {
        "hypothesis": "Wrapped hypothesis",
        "experiment_plan": "Use a real dataset and report a metric.",
    }


@pytest.mark.asyncio
async def test_run_live_fsm_fails_on_generator_provider_error(tmp_path, monkeypatch):
    db, node = _make_db_with_node(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _patch_runtime(monkeypatch, [_provider_error("generator unavailable")], [], [])

    ok = await run_live_fsm(
        node_id=node.id,
        db=db,
        config=AutoDiscoveryConfig(),
        workspace=workspace,
        domain="test domain",
        branch_path=[node],
        providers=ProviderStatus(claude_available=True, codex_available=True),
    )

    assert ok is False
    failed = db.get_node(node.id)
    assert failed.status == "failed"
    assert failed.fsm_state == "experiment_generator"
    db.close()


@pytest.mark.asyncio
async def test_run_live_fsm_analyst_provider_error_causes_failure_when_no_retries(tmp_path, monkeypatch):
    db, node = _make_db_with_node(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    cfg = AutoDiscoveryConfig()
    cfg.agents.code_attempts = 1

    research_responses = [
        _json_result({
            "hypothesis": "A real dataset signal predicts the outcome.",
            "context": "Small benchmark slice",
            "variables": ["x", "y"],
            "relationships": ["x predicts y"],
            "experiment_plan": "Use a public dataset split and report one predictive metric.",
            "cited_papers": [],
        }),
    ]
    code_responses = [_provider_error("analyst rate limited")]
    _patch_runtime(monkeypatch, research_responses, code_responses, [_backend_success])

    ok = await run_live_fsm(
        node_id=node.id,
        db=db,
        config=cfg,
        workspace=workspace,
        domain="test domain",
        branch_path=[node],
        providers=ProviderStatus(claude_available=True, codex_available=True),
    )

    assert ok is False
    failed = db.get_node(node.id)
    assert failed.status == "failed"
    assert failed.fsm_state == "experiment_analyst"
    assert failed.fsm_failure_count == 0
    db.close()


@pytest.mark.asyncio
async def test_run_live_fsm_reviewer_provider_error_does_not_auto_approve(tmp_path, monkeypatch):
    db, node = _make_db_with_node(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    cfg = AutoDiscoveryConfig()
    cfg.agents.revision_attempts = 0

    research_responses = [
        _json_result({
            "hypothesis": "A real dataset signal predicts the outcome.",
            "context": "Small benchmark slice",
            "variables": ["x", "y"],
            "relationships": ["x predicts y"],
            "experiment_plan": "Use a public dataset split and report one predictive metric.",
            "cited_papers": [],
        }),
    ]
    code_responses = [
        _json_result({
            "error": False,
            "summary": "The run produced a usable metric.",
            "key_results": {"metric": "0.9", "interpretation": "supports the hypothesis"},
            "visual_findings": "No plots generated.",
        }),
        _provider_error("reviewer malformed response"),
    ]
    _patch_runtime(monkeypatch, research_responses, code_responses, [_backend_success])

    ok = await run_live_fsm(
        node_id=node.id,
        db=db,
        config=cfg,
        workspace=workspace,
        domain="test domain",
        branch_path=[node],
        providers=ProviderStatus(claude_available=True, codex_available=True),
    )

    assert ok is False
    failed = db.get_node(node.id)
    assert failed.status == "failed"
    assert failed.fsm_state == "experiment_reviewer"
    review_text = (workspace / "experiments" / f"node_{node.id}" / "review.md").read_text()
    assert "FAILED" in review_text
    assert "reviewer malformed response" in review_text
    db.close()


@pytest.mark.asyncio
async def test_run_live_fsm_fails_on_hypothesis_provider_error(tmp_path, monkeypatch):
    db, node = _make_db_with_node(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    research_responses = [
        _json_result({
            "hypothesis": "A real dataset signal predicts the outcome.",
            "context": "Small benchmark slice",
            "variables": ["x", "y"],
            "relationships": ["x predicts y"],
            "experiment_plan": "Use a public dataset split and report one predictive metric.",
            "cited_papers": [],
        }),
        _provider_error("hypothesis generation failed"),
    ]
    code_responses = [
        _json_result({
            "error": False,
            "summary": "The run produced a usable metric.",
            "key_results": {"metric": "0.9", "interpretation": "supports the hypothesis"},
            "visual_findings": "No plots generated.",
        }),
        _json_result({
            "error": False,
            "assessment": "The experiment is valid enough for hypothesis evaluation.",
        }),
    ]
    _patch_runtime(monkeypatch, research_responses, code_responses, [_backend_success])

    ok = await run_live_fsm(
        node_id=node.id,
        db=db,
        config=AutoDiscoveryConfig(),
        workspace=workspace,
        domain="test domain",
        branch_path=[node],
        providers=ProviderStatus(claude_available=True, codex_available=True),
    )

    assert ok is False
    failed = db.get_node(node.id)
    assert failed.status == "failed"
    assert failed.fsm_state == "hypothesis_generator"
    db.close()


@pytest.mark.asyncio
async def test_run_live_fsm_fails_on_belief_provider_error(tmp_path, monkeypatch):
    db, node = _make_db_with_node(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    cfg = AutoDiscoveryConfig()
    cfg.mcts.belief_samples = 1

    research_responses = [
        _json_result({
            "hypothesis": "A real dataset signal predicts the outcome.",
            "context": "Small benchmark slice",
            "variables": ["x", "y"],
            "relationships": ["x predicts y"],
            "experiment_plan": "Use a public dataset split and report one predictive metric.",
            "cited_papers": [],
        }),
        _json_result({
            "hypothesis": "Formalized hypothesis from the experiment.",
            "context": "Same scope",
            "variables": ["x", "y"],
            "relationships": ["x predicts y"],
        }),
        _provider_error("belief prior provider error"),
    ]
    code_responses = [
        _json_result({
            "error": False,
            "summary": "The run produced a usable metric.",
            "key_results": {"metric": "0.9", "interpretation": "supports the hypothesis"},
            "visual_findings": "No plots generated.",
        }),
        _json_result({
            "error": False,
            "assessment": "The experiment is valid enough for hypothesis evaluation.",
        }),
    ]
    _patch_runtime(monkeypatch, research_responses, code_responses, [_backend_success])

    ok = await run_live_fsm(
        node_id=node.id,
        db=db,
        config=cfg,
        workspace=workspace,
        domain="test domain",
        branch_path=[node],
        providers=ProviderStatus(claude_available=True, codex_available=True),
    )

    assert ok is False
    failed = db.get_node(node.id)
    assert failed.status == "failed"
    assert failed.fsm_state == "belief_elicitation"
    db.close()


@pytest.mark.asyncio
async def test_run_live_fsm_passes_runner_system_prompt_to_backend(tmp_path, monkeypatch):
    db, node = _make_db_with_node(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    cfg = AutoDiscoveryConfig()
    cfg.mcts.belief_samples = 1

    research_responses = [
        _json_result({
            "hypothesis": "A real dataset signal predicts the outcome.",
            "context": "Small benchmark slice",
            "variables": ["x", "y"],
            "relationships": ["x predicts y"],
            "experiment_plan": "Use a public dataset split and report one predictive metric.",
            "cited_papers": [],
        }),
        _json_result({
            "hypothesis": "Formalized hypothesis from the experiment.",
            "context": "Same scope",
            "variables": ["x", "y"],
            "relationships": ["x predicts y"],
        }),
        _json_result({"believes_hypothesis": False}),
        _json_result({"believes_hypothesis": True}),
    ]
    code_responses = [
        _json_result({
            "error": False,
            "summary": "The run produced a usable metric.",
            "key_results": {"metric": "0.9", "interpretation": "supports the hypothesis"},
            "visual_findings": "No plots generated.",
        }),
        _json_result({
            "error": False,
            "assessment": "The experiment is valid enough for hypothesis evaluation.",
        }),
    ]
    backend = _patch_runtime(monkeypatch, research_responses, code_responses, [_backend_success])

    ok = await run_live_fsm(
        node_id=node.id,
        db=db,
        config=cfg,
        workspace=workspace,
        domain="test domain",
        branch_path=[node],
        providers=ProviderStatus(claude_available=True, codex_available=True),
    )

    assert ok is True
    assert backend.calls[0]["system_prompt_file"].endswith("experiment_runner.md")
    invocation_count = db.execute("SELECT COUNT(*) FROM agent_invocations WHERE node_id = ?", (node.id,)).fetchone()[0]
    assert invocation_count >= 6
    db.close()


@pytest.mark.asyncio
async def test_run_live_fsm_treats_missing_results_json_as_runner_failure(tmp_path, monkeypatch):
    db, node = _make_db_with_node(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    cfg = AutoDiscoveryConfig()
    cfg.agents.code_attempts = 1

    research_responses = [
        _json_result({
            "hypothesis": "A real dataset signal predicts the outcome.",
            "context": "Small benchmark slice",
            "variables": ["x", "y"],
            "relationships": ["x predicts y"],
            "experiment_plan": "Use a public dataset split and report one predictive metric.",
            "cited_papers": [],
        }),
    ]

    def _backend_without_results(workspace):
        (workspace / "experiment.py").write_text("print('metric = 0.9')")
        return AgentResult(raw="metric = 0.9", exit_code=0)

    _patch_runtime(monkeypatch, research_responses, [], [_backend_without_results])

    ok = await run_live_fsm(
        node_id=node.id,
        db=db,
        config=cfg,
        workspace=workspace,
        domain="test domain",
        branch_path=[node],
        providers=ProviderStatus(claude_available=True, codex_available=True),
    )

    assert ok is False
    failed = db.get_node(node.id)
    assert failed.status == "failed"
    assert failed.fsm_state == "FAIL"
    db.close()


@pytest.mark.asyncio
async def test_run_live_fsm_reuses_branch_sessions_and_updates_latest_ids(tmp_path, monkeypatch):
    db, node = _make_db_with_node(tmp_path)
    workspace = tmp_path / "exploration" / "workspaces" / "root"
    workspace.mkdir(parents=True)
    save_branch_sessions(
        tmp_path / "exploration",
        "root",
        research_claude_session_id="claude-existing",
        code_session_id="codex-existing",
        code_provider="codex",
        runner_claude_session_id="runner-existing",
    )

    cfg = AutoDiscoveryConfig()
    cfg.mcts.belief_samples = 1

    research_responses = [
        _json_result({
            "hypothesis": "A real dataset signal predicts the outcome.",
            "context": "Small benchmark slice",
            "variables": ["x", "y"],
            "relationships": ["x predicts y"],
            "experiment_plan": "Use a public dataset split and report one predictive metric.",
            "cited_papers": [],
        }, session_id="claude-new"),
        _json_result({
            "hypothesis": "Formalized hypothesis from the experiment.",
            "context": "Same scope",
            "variables": ["x", "y"],
            "relationships": ["x predicts y"],
        }, session_id="claude-new"),
        _json_result({"believes_hypothesis": False}, session_id="claude-new"),
        _json_result({"believes_hypothesis": True}, session_id="claude-new"),
    ]
    code_responses = [
        _json_result({
            "error": False,
            "summary": "The run produced a usable metric.",
            "key_results": {"metric": "0.9", "interpretation": "supports the hypothesis"},
            "visual_findings": "No plots generated.",
        }, session_id="codex-new"),
        _json_result({
            "error": False,
            "assessment": "The experiment is valid enough for hypothesis evaluation.",
        }, session_id="codex-new"),
    ]
    def _backend_success_with_session(workspace):
        (workspace / "results.json").write_text(json.dumps({
            "code": "print('metric = 0.9')",
            "stdout": "metric = 0.9",
            "metrics": {"metric": 0.9},
            "error": False,
        }))
        return AgentResult(raw="metric = 0.9", exit_code=0, session_id="runner-new")

    backend = _patch_runtime(monkeypatch, research_responses, code_responses, [_backend_success_with_session])

    ok = await run_live_fsm(
        node_id=node.id,
        db=db,
        config=cfg,
        workspace=workspace,
        domain="test domain",
        branch_path=[node],
        providers=ProviderStatus(claude_available=True, codex_available=True),
    )

    assert ok is True
    assert backend.research_agent.calls[0]["session_id"] == "claude-existing"
    assert backend.research_agent.calls[0]["resume_session"] is True
    # Codex agents use fresh sessions (no resume) because exec resume
    # doesn't support -o for clean output capture.
    assert "session_id" not in backend.code_agent.calls[0]
    assert backend.calls[0]["session_id"] == "runner-existing"
    assert backend.research_agent.calls[1]["session_id"] == "claude-new"
    assert backend.research_agent.calls[1]["resume_session"] is True
    assert backend.research_agent.calls[2]["session_id"] == "claude-new"
    assert backend.research_agent.calls[2]["resume_session"] is True
    assert backend.research_agent.calls[2]["fork_session"] is True
    assert backend.research_agent.calls[3]["session_id"] == "claude-new"
    assert backend.research_agent.calls[3]["resume_session"] is True
    assert backend.research_agent.calls[3]["fork_session"] is True

    updated_node = db.get_node(node.id)
    assert updated_node.claude_session_id == "claude-new"
    assert updated_node.codex_session_id == "codex-new"
    sessions = load_branch_sessions(tmp_path / "exploration", "root")
    assert sessions["research_claude_session_id"] == "claude-new"
    assert sessions["code_session_id"] == "codex-new"
    assert sessions["code_provider"] == "codex"
    assert sessions["runner_claude_session_id"] == "runner-new"
    db.close()
