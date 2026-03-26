import pytest
from surprisal.agents.experiment_container import ExperimentContainer
from surprisal.config import SandboxConfig, CredentialsConfig


def test_build_command_with_gpu():
    config = SandboxConfig(gpu=True, image="surprisal-gpu:latest", memory_limit="16g", cpu_limit="4")
    creds = CredentialsConfig()
    container = ExperimentContainer(config=config, credentials=creds)
    cmd = container.build_run_command("/tmp/ws", "/tmp/mcp.json", "test prompt")
    assert "--gpus=all" in cmd
    assert "--memory" in cmd
    assert "16g" in cmd
    assert "--cpus" in cmd
    assert "4" in cmd
    assert "-v" in cmd
    assert "/tmp/ws:/work:rw" in cmd
    assert "/tmp/mcp.json:/etc/surprisal/mcp.json:ro" in cmd
    assert "surprisal-gpu:latest" in cmd
    assert "claude" in cmd
    assert "--dangerously-skip-permissions" in cmd
    assert "--mcp-config" in cmd


def test_build_command_without_gpu():
    config = SandboxConfig(gpu=False, image="surprisal-sandbox:latest")
    creds = CredentialsConfig()
    container = ExperimentContainer(config=config, credentials=creds)
    cmd = container.build_run_command("/tmp/ws", "/tmp/mcp.json", "test prompt")
    assert "--gpus=all" not in cmd


def test_build_command_with_wandb_creds():
    config = SandboxConfig()
    creds = CredentialsConfig(wandb_api_key="test-wandb-key")
    container = ExperimentContainer(config=config, credentials=creds)
    cmd = container.build_run_command("/tmp/ws", "/tmp/mcp.json", "test prompt")
    env_idx = cmd.index("-e")
    assert "WANDB_API_KEY=test-wandb-key" in cmd[env_idx + 1]


def test_build_command_with_hf_creds():
    config = SandboxConfig()
    creds = CredentialsConfig(hf_token="hf_test_token")
    container = ExperimentContainer(config=config, credentials=creds)
    cmd = container.build_run_command("/tmp/ws", "/tmp/mcp.json", "test prompt")
    env_args = [cmd[i + 1] for i, x in enumerate(cmd) if x == "-e"]
    assert any("HF_TOKEN=hf_test_token" in arg for arg in env_args)


def test_build_command_no_creds_no_env():
    config = SandboxConfig()
    creds = CredentialsConfig()
    container = ExperimentContainer(config=config, credentials=creds)
    cmd = container.build_run_command("/tmp/ws", "/tmp/mcp.json", "test prompt")
    assert "-e" not in cmd


def test_is_infra_error():
    assert ExperimentContainer.is_infra_error(125) is True
    assert ExperimentContainer.is_infra_error(126) is True
    assert ExperimentContainer.is_infra_error(127) is True
    assert ExperimentContainer.is_infra_error(1) is False
    assert ExperimentContainer.is_infra_error(0) is False
