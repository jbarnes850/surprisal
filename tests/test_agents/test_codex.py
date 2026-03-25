from surprisal.agents.codex import CodexAgent


def test_codex_builds_correct_command():
    agent = CodexAgent(model="gpt-5.4")
    cmd = agent.build_command(prompt="write code")
    assert cmd[0] == "codex"
    assert "exec" in cmd
    assert "--full-auto" in cmd
    assert 'model="gpt-5.4"' in cmd
    assert cmd[-1] == "write code"


def test_codex_command_with_output_file():
    agent = CodexAgent()
    cmd = agent.build_command(prompt="test", output_file="/tmp/out.txt")
    assert "-o" in cmd
    assert "/tmp/out.txt" in cmd
