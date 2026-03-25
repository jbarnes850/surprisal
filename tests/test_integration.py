"""Integration tests: full single-node MCTS lifecycle and CLI end-to-end flows."""

import json
import pytest
from click.testing import CliRunner
from surprisal.db import Database
from surprisal.models import Node
from surprisal.mcts import select_node, backpropagate
from surprisal.bayesian import compute_surprisal
from surprisal.cli import main


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
        data = json.loads(r.output)

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
        runner = CliRunner()
        runner.invoke(main, ["init", "--domain", "test", "--seed", "hypothesis"])
        r = runner.invoke(main, ["explore", "--budget", "2", "--concurrency", "1"])
        assert r.exit_code == 0, f"explore failed: {r.output}"
        # Should have expanded nodes (placeholder FSM)
        output = json.loads(r.output)
        assert output["iterations"] >= 1

    def test_export_after_explore(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SURPRISAL_HOME", str(tmp_path))
        runner = CliRunner()
        runner.invoke(main, ["init", "--domain", "test", "--seed", "hypothesis"])
        runner.invoke(main, ["explore", "--budget", "3", "--concurrency", "1"])
        r = runner.invoke(main, ["export", "--format", "json"])
        assert r.exit_code == 0, f"export failed: {r.output}"

    def test_prune_after_explore(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SURPRISAL_HOME", str(tmp_path))
        runner = CliRunner()
        runner.invoke(main, ["init", "--domain", "test", "--seed", "hypothesis"])
        runner.invoke(main, ["explore", "--budget", "3", "--concurrency", "1"])
        r = runner.invoke(main, ["prune", "--dry-run"])
        assert r.exit_code == 0, f"prune failed: {r.output}"
