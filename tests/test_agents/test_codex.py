from autodiscovery.agents.codex import CodexAgent


def test_codex_builds_correct_command():
    agent = CodexAgent(model="gpt-5.4")
    cmd = agent.build_command(prompt="write code", cwd="/tmp/workspace")
    assert cmd[0] == "codex"
    assert "-q" in cmd
    assert "--full-auto" in cmd
    assert "--model" in cmd
    assert "gpt-5.4" in cmd
    assert "-f" in cmd
    assert "/tmp/workspace" in cmd
    assert cmd[-1] == "write code"


def test_codex_command_without_cwd():
    agent = CodexAgent()
    cmd = agent.build_command(prompt="test")
    assert "-f" not in cmd
