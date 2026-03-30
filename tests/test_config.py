import pytest
from surprisal.config import (
    AUTO_IMAGE,
    AutoDiscoveryConfig,
    DEFAULT_CPU_IMAGE,
    DEFAULT_GPU_IMAGE,
    load_config,
    resolve_sandbox_image,
    save_config,
)


def test_default_config_has_expected_values():
    cfg = AutoDiscoveryConfig()
    assert cfg.general.default_budget == 100
    assert cfg.general.default_concurrency == 2
    assert cfg.mcts.c_explore == pytest.approx(1.414, abs=0.001)
    assert cfg.mcts.k_progressive == 1.0
    assert cfg.mcts.alpha_progressive == 0.5
    assert cfg.mcts.max_depth == 30
    assert cfg.mcts.virtual_loss == 2
    assert cfg.mcts.dedup_interval == 50
    assert cfg.belief.provider == "claude"
    assert cfg.belief.samples == 30
    assert cfg.belief.kl_scale == 5.0
    assert cfg.belief.evidence_weight == 2.0
    assert cfg.agents.claude_model == "opus"
    assert cfg.agents.codex_model == "gpt-5.4"
    assert cfg.agents.max_turns == 20
    assert cfg.agents.code_attempts == 3
    assert cfg.agents.revision_attempts == 1
    assert cfg.sandbox.memory_limit == "16g"
    assert cfg.sandbox.cpu_limit == "4"
    assert cfg.sandbox.timeout == 1800
    assert cfg.sandbox.network is True
    assert not hasattr(cfg, "predictor")


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


def test_generator_timeout_default():
    cfg = AutoDiscoveryConfig()
    assert cfg.agents.generator_timeout == 180


def test_config_set_invalid_key_raises():
    cfg = AutoDiscoveryConfig()
    with pytest.raises(KeyError):
        cfg.set("nonexistent.key", "value")


def test_sandbox_config_new_defaults():
    cfg = AutoDiscoveryConfig()
    assert cfg.sandbox.backend == "auto"
    assert cfg.sandbox.image == AUTO_IMAGE
    assert cfg.sandbox.gpu is True
    assert cfg.sandbox.memory_limit == "16g"
    assert cfg.sandbox.cpu_limit == "4"
    assert cfg.sandbox.timeout == 1800
    assert cfg.sandbox.network is True
    assert cfg.sandbox.hf_flavor == "a10g-small"
    assert cfg.sandbox.hf_timeout == "2h"


def test_credentials_config_defaults():
    cfg = AutoDiscoveryConfig()
    assert cfg.credentials.wandb_api_key == ""
    assert cfg.credentials.hf_token == ""


def test_credentials_config_round_trip(tmp_path):
    cfg = AutoDiscoveryConfig()
    cfg.credentials.wandb_api_key = "test-key-123"
    cfg.credentials.hf_token = "hf_test_456"
    path = tmp_path / "config.toml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.credentials.wandb_api_key == "test-key-123"
    assert loaded.credentials.hf_token == "hf_test_456"


def test_load_config_ignores_removed_sections(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[belief]
samples = 12

[predictor]
enabled = true
lambda_weight = 0.5
"""
    )
    loaded = load_config(path)
    assert loaded.belief.samples == 12
    assert not hasattr(loaded, "predictor")


def test_config_set_credentials():
    cfg = AutoDiscoveryConfig()
    cfg.set("credentials.wandb_api_key", "my-key")
    assert cfg.credentials.wandb_api_key == "my-key"


def test_config_set_sandbox_gpu():
    cfg = AutoDiscoveryConfig()
    cfg.set("sandbox.gpu", "false")
    assert cfg.sandbox.gpu is False


def test_resolve_sandbox_image_auto_cpu():
    assert resolve_sandbox_image(AUTO_IMAGE, gpu_enabled=False) == DEFAULT_CPU_IMAGE


def test_resolve_sandbox_image_auto_gpu():
    assert resolve_sandbox_image(AUTO_IMAGE, gpu_enabled=True) == DEFAULT_GPU_IMAGE


def test_resolve_sandbox_image_explicit():
    assert resolve_sandbox_image("custom-image:latest", gpu_enabled=False) == "custom-image:latest"
