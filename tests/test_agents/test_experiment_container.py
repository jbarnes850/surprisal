import pytest

from surprisal.agents.experiment_container import ExperimentContainer
from surprisal.config import CredentialsConfig, SandboxConfig


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
    config = SandboxConfig(gpu=False, image="surprisal-cpu:latest")
    creds = CredentialsConfig()
    container = ExperimentContainer(config=config, credentials=creds)
    cmd = container.build_run_command("/tmp/ws", "/tmp/mcp.json", "test prompt")
    assert "--gpus=all" not in cmd
    assert "surprisal-cpu:latest" in cmd


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


def test_build_command_forwards_runner_auth_env(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "secret-token")
    config = SandboxConfig()
    creds = CredentialsConfig()
    container = ExperimentContainer(config=config, credentials=creds)
    cmd = container.build_run_command("/tmp/ws", "/tmp/mcp.json", "test prompt")
    env_args = [cmd[i + 1] for i, x in enumerate(cmd) if x == "-e"]
    assert "CLAUDE_CODE_OAUTH_TOKEN=secret-token" in env_args


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
    assert "/tmp/claude-home:/home/surprisal:rw" in cmd
    assert "HOME=/home/surprisal" in cmd


def test_prepare_runner_claude_home_seeds_branch_state(tmp_path, monkeypatch):
    host_home = tmp_path / "host-home"
    host_claude = host_home / ".claude"
    host_claude.mkdir(parents=True)
    (host_claude / ".credentials.json").write_text('{"token":"test"}')
    (host_claude / "settings.json").write_text("{}")
    (host_home / ".claude.json").write_text('{"projects":{}}')

    workspace = tmp_path / "workspaces" / "branch-a" / "experiments" / "node-1"
    workspace.mkdir(parents=True)

    monkeypatch.setattr(
        "surprisal.agents.experiment_container.Path.home",
        staticmethod(lambda: host_home),
    )

    runner_home = ExperimentContainer.prepare_runner_claude_home(workspace)

    assert runner_home == tmp_path / "workspaces" / "branch-a" / ".claude-runner"
    assert (runner_home / ".claude" / ".credentials.json").read_text() == '{"token":"test"}'
    assert (runner_home / ".claude" / "settings.json").read_text() == "{}"
    assert (runner_home / ".claude.json").read_text() == '{"projects":{}}'


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
    async def _fake_ensure_image(_tag, progress_callback=None):
        return True, ""

    monkeypatch.setattr(
        ExperimentContainer,
        "ensure_image",
        staticmethod(_fake_ensure_image),
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


@pytest.mark.asyncio
async def test_ensure_image_emits_cpu_preflight_message(monkeypatch):
    messages = []

    async def _fake_image_exists(_tag):
        return False

    async def _fake_build_image(_dockerfile_path, tag="surprisal-cpu:latest"):
        assert tag == "surprisal-cpu:latest"
        return True

    monkeypatch.setattr(ExperimentContainer, "image_exists", staticmethod(_fake_image_exists))
    monkeypatch.setattr(ExperimentContainer, "build_image", staticmethod(_fake_build_image))

    ok, error = await ExperimentContainer.ensure_image(
        "surprisal-cpu:latest",
        progress_callback=messages.append,
    )

    assert ok is True
    assert error == ""
    assert messages[0] == "CPU image missing, building once for local execution."
    assert any("Runner: building `surprisal-cpu:latest`" in message for message in messages)


def test_is_infra_error():
    assert ExperimentContainer.is_infra_error(125) is True
    assert ExperimentContainer.is_infra_error(126) is True
    assert ExperimentContainer.is_infra_error(127) is True
    assert ExperimentContainer.is_infra_error(1) is False
    assert ExperimentContainer.is_infra_error(0) is False
