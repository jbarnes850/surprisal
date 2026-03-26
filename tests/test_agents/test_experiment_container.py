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


def test_build_command_with_system_prompt_and_network_disabled():
    config = SandboxConfig(network=False)
    creds = CredentialsConfig()
    container = ExperimentContainer(config=config, credentials=creds)
    cmd = container.build_run_command(
        "/tmp/ws",
        "/tmp/mcp.json",
        "test prompt",
        system_prompt="You are the runner.",
    )
    assert "--network" in cmd
    assert "none" in cmd
    assert "--system-prompt" in cmd
    assert "You are the runner." in cmd


def test_build_command_with_resume_session():
    config = SandboxConfig()
    creds = CredentialsConfig()
    container = ExperimentContainer(config=config, credentials=creds)
    cmd = container.build_run_command(
        "/tmp/ws",
        "/tmp/mcp.json",
        "test prompt",
        session_id="claude-session-123",
        claude_home="/tmp/claude-home",
    )
    assert "--resume" in cmd
    assert "claude-session-123" in cmd
    assert "/tmp/claude-home:/root/.claude:rw" in cmd
    assert "HOME=/root" in cmd


def test_prepare_runner_claude_home_seeds_branch_state(tmp_path, monkeypatch):
    host_home = tmp_path / "host-home"
    host_claude = host_home / ".claude"
    host_claude.mkdir(parents=True)
    (host_claude / ".credentials.json").write_text('{"token":"test"}')
    (host_claude / "settings.json").write_text("{}")

    workspace = tmp_path / "workspaces" / "branch-a" / "experiments" / "node-1"
    workspace.mkdir(parents=True)

    monkeypatch.setattr(
        "surprisal.agents.experiment_container.Path.home",
        staticmethod(lambda: host_home),
    )

    runner_home = ExperimentContainer.prepare_runner_claude_home(workspace)

    assert runner_home == tmp_path / "workspaces" / "branch-a" / ".claude-runner"
    assert (runner_home / ".credentials.json").read_text() == '{"token":"test"}'
    assert (runner_home / "settings.json").read_text() == "{}"


@pytest.mark.asyncio
async def test_execute_resume_without_host_claude_home_cleans_up_mcp_config(tmp_path, monkeypatch):
    config = SandboxConfig()
    creds = CredentialsConfig()
    container = ExperimentContainer(config=config, credentials=creds)
    workspace = tmp_path / "workspaces" / "branch-a" / "experiments" / "node-1"
    workspace.mkdir(parents=True)

    mcp_config_path = tmp_path / "mcp.json"

    def _fake_write_mcp_config(_credentials):
        mcp_config_path.write_text("{}")
        return str(mcp_config_path)

    monkeypatch.setattr(
        "surprisal.agents.experiment_container.write_mcp_config",
        _fake_write_mcp_config,
    )
    monkeypatch.setattr(
        "surprisal.agents.experiment_container.Path.home",
        staticmethod(lambda: tmp_path / "missing-home"),
    )

    result = await container.execute(
        experiment_prompt="test prompt",
        workspace=workspace,
        config=config,
        session_id="runner-session-123",
    )

    assert result.exit_code == 1
    assert "session persistence unavailable" in result.raw
    assert not mcp_config_path.exists()


def test_is_infra_error():
    assert ExperimentContainer.is_infra_error(125) is True
    assert ExperimentContainer.is_infra_error(126) is True
    assert ExperimentContainer.is_infra_error(127) is True
    assert ExperimentContainer.is_infra_error(1) is False
    assert ExperimentContainer.is_infra_error(0) is False
