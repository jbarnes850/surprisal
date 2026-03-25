import pytest
from pathlib import Path
from autodiscovery.config import AutoDiscoveryConfig, load_config, save_config


def test_default_config_has_expected_values():
    cfg = AutoDiscoveryConfig()
    assert cfg.general.default_budget == 100
    assert cfg.general.default_concurrency == 2
    assert cfg.mcts.c_explore == pytest.approx(1.414, abs=0.001)
    assert cfg.mcts.k_progressive == 1.0
    assert cfg.mcts.alpha_progressive == 0.5
    assert cfg.mcts.max_depth == 30
    assert cfg.mcts.belief_samples == 30
    assert cfg.mcts.virtual_loss == 2
    assert cfg.mcts.dedup_interval == 50
    assert cfg.agents.claude_model == "opus"
    assert cfg.agents.codex_model == "gpt-5.4"
    assert cfg.agents.max_turns == 20
    assert cfg.agents.code_attempts == 6
    assert cfg.agents.revision_attempts == 1
    assert cfg.sandbox.memory_limit == "2g"
    assert cfg.sandbox.cpu_limit == "1.5"
    assert cfg.sandbox.timeout == 600
    assert cfg.sandbox.network is False
    assert cfg.predictor.enabled is False


def test_config_round_trip(tmp_path):
    cfg = AutoDiscoveryConfig()
    path = tmp_path / "config.toml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.mcts.c_explore == cfg.mcts.c_explore
    assert loaded.agents.claude_model == cfg.agents.claude_model


def test_load_config_returns_defaults_when_missing(tmp_path):
    path = tmp_path / "nonexistent.toml"
    cfg = load_config(path)
    assert cfg.general.default_budget == 100


def test_config_set_value():
    cfg = AutoDiscoveryConfig()
    cfg.set("mcts.c_explore", "2.0")
    assert cfg.mcts.c_explore == pytest.approx(2.0)


def test_config_set_invalid_key_raises():
    cfg = AutoDiscoveryConfig()
    with pytest.raises(KeyError):
        cfg.set("nonexistent.key", "value")
