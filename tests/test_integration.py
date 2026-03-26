"""Integration tests: full single-node MCTS lifecycle and CLI end-to-end flows."""

import json
from click.testing import CliRunner
from surprisal.models import Node
from surprisal.mcts import select_node, backpropagate
from surprisal.bayesian import compute_surprisal
from surprisal.cli import main
from surprisal.providers import LiteratureStatus, ProviderStatus


def _patch_cli_explore(monkeypatch, *, iterations: int = 1) -> None:
    async def _fake_detect_providers():
        return ProviderStatus(claude_available=True, codex_available=True)

    async def _fake_detect_literature_provider():
        return LiteratureStatus(provider="huggingface")

    async def _fake_run_exploration(**kwargs):
        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback("Starting exploration with budget 1 and concurrency 1.")
            progress_callback("Node child-123: starting stage `experiment_generator`.")
        return {
            "status": "completed",
            "iterations": iterations,
            "nodes_total": 2,
            "surprisals_found": 0,
        }

    monkeypatch.setattr("surprisal.providers.detect_providers", _fake_detect_providers)
    monkeypatch.setattr("surprisal.providers.detect_literature_provider", _fake_detect_literature_provider)
    monkeypatch.setattr("surprisal.cli.run_exploration", _fake_run_exploration)


class TestSingleNodeLifecycle:
    """Test: select root -> expand -> compute surprisal -> backpropagate."""

    def test_selection_returns_root_for_fresh_tree(self, tmp_db):
        tmp_db.insert_node(Node(id="root", exploration_id="e", hypothesis="Root", visit_count=1))
        selected = select_node(tmp_db, "root", c_explore=1.414, k=1.0, alpha=0.5)
        assert selected == "root"

    def test_child_creation_and_surprisal(self, tmp_db):
        tmp_db.insert_node(Node(id="root", exploration_id="e", hypothesis="Root", visit_count=1))
        # Simulate expansion: create child
        child = Node(id="child1", exploration_id="e", hypothesis="Child hypothesis",
                     parent_id="root", depth=1, status="expanding")
        tmp_db.insert_node(child)

        # Compute surprisal from mock belief samples (belief shifted)
        result = compute_surprisal(k_prior=25, k_post=5, n=30)
        assert result.surprisal == 1
        assert result.belief_shifted is True

        # Update node with surprisal
        tmp_db.update_node("child1",
            status="verified",
            bayesian_surprise=result.bayesian_surprise,
            belief_shifted=result.belief_shifted,
            prior_alpha=result.prior_alpha,
            prior_beta=result.prior_beta,
            posterior_alpha=result.posterior_alpha,
            posterior_beta=result.posterior_beta,
            k_prior=25, k_post=5,
            virtual_loss=0,
        )

        # Backpropagate
        backpropagate(tmp_db, "child1", result.surprisal)

        # Verify tree state
        assert tmp_db.get_node("child1").visit_count == 1
        assert tmp_db.get_node("child1").surprisal_sum == 1.0
        assert tmp_db.get_node("root").visit_count == 2  # was 1, now 2
        assert tmp_db.get_node("root").surprisal_sum == 1.0

    def test_no_surprisal_path(self, tmp_db):
        tmp_db.insert_node(Node(id="root", exploration_id="e", hypothesis="Root", visit_count=1))
        child = Node(id="child2", exploration_id="e", hypothesis="No shift",
                     parent_id="root", depth=1, status="expanding")
        tmp_db.insert_node(child)

        result = compute_surprisal(k_prior=20, k_post=22, n=30)
        assert result.surprisal == 0
        assert result.belief_shifted is False

        tmp_db.update_node("child2", status="verified", virtual_loss=0,
                           bayesian_surprise=result.bayesian_surprise,
                           belief_shifted=result.belief_shifted)
        backpropagate(tmp_db, "child2", result.surprisal)

        assert tmp_db.get_node("child2").visit_count == 1
        assert tmp_db.get_node("child2").surprisal_sum == 0.0
        assert tmp_db.get_node("root").visit_count == 2
        assert tmp_db.get_node("root").surprisal_sum == 0.0

    def test_progressive_widening_after_visits(self, tmp_db):
        """After 4 visits to root, root can have 2 children."""
        tmp_db.insert_node(Node(id="root", exploration_id="e", hypothesis="Root", visit_count=4))
        tmp_db.insert_node(Node(id="c1", exploration_id="e", hypothesis="Child 1",
                                parent_id="root", depth=1, visit_count=1))
        # With 4 visits, max_children = floor(1.0 * 4^0.5) = 2
        # Only 1 child exists -> can add another -> select returns root
        selected = select_node(tmp_db, "root", c_explore=1.414, k=1.0, alpha=0.5)
        assert selected == "root"


class TestCLIIntegration:
    """Test CLI commands work end-to-end."""

    def test_init_then_status(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SURPRISAL_HOME", str(tmp_path))
        runner = CliRunner()

        # Init
        r = runner.invoke(main, ["init", "--domain", "integration test", "--seed", "test h", "--json"])
        assert r.exit_code == 0, f"init failed: {r.output}"
        json.loads(r.output)

        # Status
        r = runner.invoke(main, ["status", "--json"])
        assert r.exit_code == 0, f"status failed: {r.output}"
        status = json.loads(r.output)
        assert status["domain"] == "integration test"
        assert status["nodes_total"] >= 1

    def test_init_then_explore_dryrun(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SURPRISAL_HOME", str(tmp_path))
        runner = CliRunner()
        runner.invoke(main, ["init", "--domain", "test", "--seed", "hypothesis"])
        r = runner.invoke(main, ["explore", "--dry-run"])
        assert r.exit_code == 0, f"explore dry-run failed: {r.output}"
        assert "Would expand" in r.output

    def test_init_then_explore_budget(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SURPRISAL_HOME", str(tmp_path))
        _patch_cli_explore(monkeypatch, iterations=2)
        runner = CliRunner()
        runner.invoke(main, ["init", "--domain", "test", "--seed", "hypothesis"])
        r = runner.invoke(main, ["explore", "--budget", "2", "--concurrency", "1", "--json"])
        assert r.exit_code == 0, f"explore failed: {r.output}"
        output = json.loads(r.output)
        assert output["iterations"] >= 1

    def test_explore_streams_progress_messages(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SURPRISAL_HOME", str(tmp_path))
        _patch_cli_explore(monkeypatch, iterations=1)
        runner = CliRunner()
        runner.invoke(main, ["init", "--domain", "test", "--seed", "hypothesis"])
        r = runner.invoke(main, ["explore", "--budget", "1", "--concurrency", "1"])
        assert r.exit_code == 0, f"explore failed: {r.output}"
        assert "Progress: Starting exploration with budget 1 and concurrency 1." in r.output
        assert "Progress: Node child-123: starting stage `experiment_generator`." in r.output

    def test_resume_dry_run_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SURPRISAL_HOME", str(tmp_path))
        runner = CliRunner()
        init = runner.invoke(main, ["init", "--domain", "test", "--seed", "hypothesis", "--json"])
        exp_id = json.loads(init.output)["exploration_id"]
        r = runner.invoke(main, ["resume", exp_id, "--dry-run", "--json"])
        assert r.exit_code == 0, f"resume failed: {r.output}"
        output = json.loads(r.output)
        assert output["status"] == "dry_run"

    def test_export_after_explore(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SURPRISAL_HOME", str(tmp_path))
        _patch_cli_explore(monkeypatch, iterations=1)
        runner = CliRunner()
        runner.invoke(main, ["init", "--domain", "test", "--seed", "hypothesis"])
        runner.invoke(main, ["explore", "--budget", "3", "--concurrency", "1"])
        r = runner.invoke(main, ["export", "--format", "json"])
        assert r.exit_code == 0, f"export failed: {r.output}"

    def test_prune_after_explore(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SURPRISAL_HOME", str(tmp_path))
        _patch_cli_explore(monkeypatch, iterations=1)
        runner = CliRunner()
        runner.invoke(main, ["init", "--domain", "test", "--seed", "hypothesis"])
        runner.invoke(main, ["explore", "--budget", "3", "--concurrency", "1"])
        r = runner.invoke(main, ["prune", "--dry-run"])
        assert r.exit_code == 0, f"prune failed: {r.output}"

    def test_explore_smoke_creates_verified_child_node(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SURPRISAL_HOME", str(tmp_path))

        async def _fake_detect_providers():
            return ProviderStatus(claude_available=True, codex_available=True)

        async def _fake_detect_literature_provider():
            return LiteratureStatus(provider="huggingface")

        async def _fake_run_live_fsm(node_id, db, progress_callback=None, **kwargs):
            if progress_callback is not None:
                progress_callback(f"Node {node_id}: starting stage `experiment_generator`.")
                progress_callback(f"Node {node_id}: starting stage `experiment_runner`.")
            db.update_node(
                node_id,
                hypothesis="Verified smoke hypothesis",
                status="verified",
                virtual_loss=0,
                belief_shifted=False,
                bayesian_surprise=0.0,
            )
            return True

        monkeypatch.setattr("surprisal.providers.detect_providers", _fake_detect_providers)
        monkeypatch.setattr("surprisal.providers.detect_literature_provider", _fake_detect_literature_provider)
        monkeypatch.setattr("surprisal.fsm_runner.run_live_fsm", _fake_run_live_fsm)

        runner = CliRunner()
        runner.invoke(main, ["init", "--domain", "smoke test", "--seed", "root hypothesis"])
        explore = runner.invoke(main, ["explore", "--budget", "1", "--concurrency", "1"])
        assert explore.exit_code == 0, f"explore failed: {explore.output}"
        assert "Progress: Node " in explore.output
        assert "starting stage `experiment_runner`" in explore.output

        status = runner.invoke(main, ["status", "--json"])
        assert status.exit_code == 0, f"status failed: {status.output}"
        payload = json.loads(status.output)
        assert payload["nodes_total"] == 2
        assert payload["verified"] == 2
        assert payload["failed"] == 0
