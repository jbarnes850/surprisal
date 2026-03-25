import json
from click.testing import CliRunner
from autodiscovery.cli import main


def test_init_creates_exploration(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTODISCOVERY_HOME", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(main, ["init", "--domain", "test domain", "--seed", "test hypothesis", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "exploration_id" in data
    assert "root_node_id" in data


def test_init_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTODISCOVERY_HOME", str(tmp_path))
    runner = CliRunner()
    r1 = runner.invoke(main, ["init", "--domain", "test", "--seed", "h", "--json"])
    r2 = runner.invoke(main, ["init", "--domain", "test", "--seed", "h", "--json"])
    assert json.loads(r1.output)["exploration_id"] == json.loads(r2.output)["exploration_id"]


def test_status_shows_exploration(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTODISCOVERY_HOME", str(tmp_path))
    runner = CliRunner()
    runner.invoke(main, ["init", "--domain", "test domain", "--seed", "h"])
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "test domain" in result.output


def test_config_show(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTODISCOVERY_HOME", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(main, ["config", "--show"])
    assert result.exit_code == 0
    assert "c_explore" in result.output


def test_export_no_exploration_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTODISCOVERY_HOME", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(main, ["export"])
    assert result.exit_code != 0
    assert "No exploration" in result.output


def test_prune_dry_run(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTODISCOVERY_HOME", str(tmp_path))
    runner = CliRunner()
    runner.invoke(main, ["init", "--domain", "test", "--seed", "h"])
    result = runner.invoke(main, ["prune", "--dry-run"])
    assert result.exit_code == 0


def test_explore_dry_run(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTODISCOVERY_HOME", str(tmp_path))
    runner = CliRunner()
    runner.invoke(main, ["init", "--domain", "test", "--seed", "h"])
    result = runner.invoke(main, ["explore", "--dry-run"])
    assert result.exit_code == 0
    assert "Would expand" in result.output
