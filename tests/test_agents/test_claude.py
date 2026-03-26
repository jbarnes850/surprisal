import pytest
import json
from surprisal.agents.base import AgentResult
from surprisal.agents.claude import ClaudeAgent


def test_claude_agent_builds_correct_command():
    agent = ClaudeAgent(model="opus", max_turns=20)
    cmd = agent.build_command(
        prompt="test prompt",
        system_prompt_file="/tmp/prompt.md",
        output_format="json",
    )
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert "--dangerously-skip-permissions" in cmd
    assert "--model" in cmd
    assert "opus" in cmd
    assert "--output-format" in cmd
    assert "json" in cmd
    assert "--max-turns" in cmd
    assert "--system-prompt-file" in cmd


def test_claude_agent_builds_command_with_session():
    agent = ClaudeAgent(model="opus", max_turns=20)
    cmd = agent.build_command(prompt="test", system_prompt_file="/tmp/p.md", session_id="abc-123")
    assert "--session-id" in cmd
    assert "abc-123" in cmd


def test_claude_agent_builds_command_with_resume():
    agent = ClaudeAgent(model="opus", max_turns=20)
    cmd = agent.build_command(prompt="test", system_prompt_file="/tmp/p.md", resume_session="abc-123")
    assert "--resume" in cmd
    assert "abc-123" in cmd


def test_claude_agent_builds_command_with_fork():
    agent = ClaudeAgent(model="opus", max_turns=20)
    cmd = agent.build_command(prompt="test", system_prompt_file="/tmp/p.md", resume_session="abc-123", fork_session=True)
    assert "--resume" in cmd
    assert "--fork-session" in cmd


def test_claude_agent_builds_command_with_json_schema():
    agent = ClaudeAgent(model="opus", max_turns=20)
    schema = {"type": "object", "properties": {"hypothesis": {"type": "string"}}}
    cmd = agent.build_command(prompt="test", system_prompt_file="/tmp/p.md", json_schema=schema)
    assert "--json-schema" in cmd


def test_agent_result_from_json():
    raw = json.dumps({"result": "some text", "session_id": "s1"})
    result = AgentResult.from_raw(raw, exit_code=0)
    assert result.exit_code == 0
    assert result.parsed["result"] == "some text"


def test_agent_result_from_non_json():
    result = AgentResult.from_raw("not json", exit_code=0)
    assert result.parsed is None
    assert result.raw == "not json"


def test_fork_without_resume_is_ignored():
    agent = ClaudeAgent(model="opus", max_turns=20)
    cmd = agent.build_command(prompt="test", fork_session=True)  # no resume_session
    assert "--fork-session" not in cmd


def test_claude_agent_builds_command_with_extra_args():
    agent = ClaudeAgent(model="opus", max_turns=20)
    cmd = agent.build_command(
        prompt="test",
        extra_args=["--mcp-config", "/tmp/mcp.json"],
    )
    assert "--mcp-config" in cmd
    assert "/tmp/mcp.json" in cmd


def test_extra_args_none_does_not_add_flags():
    agent = ClaudeAgent(model="opus", max_turns=20)
    cmd = agent.build_command(prompt="test")
    base_len = len(cmd)
    cmd2 = agent.build_command(prompt="test", extra_args=None)
    assert len(cmd2) == base_len
