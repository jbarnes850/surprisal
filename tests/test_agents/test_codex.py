from surprisal.agents.codex import CodexAgent, extract_thread_id, extract_message_content


def test_codex_builds_correct_command():
    agent = CodexAgent(model="gpt-5.4")
    cmd = agent.build_command(prompt="write code")
    assert cmd[0] == "codex"
    assert "exec" in cmd
    assert "--full-auto" in cmd
    assert "--json" in cmd
    assert "--skip-git-repo-check" in cmd
    assert 'model="gpt-5.4"' in cmd
    assert "--ephemeral" not in cmd
    assert cmd[-1] == "write code"


def test_codex_command_with_output_file():
    agent = CodexAgent()
    cmd = agent.build_command(prompt="test", output_file="/tmp/out.txt")
    assert "-o" in cmd
    assert "/tmp/out.txt" in cmd


def test_codex_compose_prompt_includes_system_prompt(tmp_path):
    prompt_file = tmp_path / "system.md"
    prompt_file.write_text("You are the reviewer.")
    agent = CodexAgent()
    composed = agent.compose_prompt("Check the run.", system_prompt_file=str(prompt_file))
    assert "You are the reviewer." in composed
    assert "User task:\nCheck the run." in composed


def test_codex_build_command_with_output_schema_and_extra_args():
    agent = CodexAgent()
    cmd = agent.build_command(
        prompt="Return JSON",
        output_schema_file="/tmp/schema.json",
        extra_args=["--json"],
    )
    assert "--output-schema" in cmd
    assert "/tmp/schema.json" in cmd
    assert "--json" in cmd


def test_codex_build_command_for_resume():
    agent = CodexAgent()
    cmd = agent.build_command(
        prompt="Resume this thread",
        session_id="thread-123",
    )
    assert cmd[:3] == ["codex", "exec", "resume"]
    assert "thread-123" in cmd


def test_extract_thread_id_from_jsonl():
    jsonl = "\n".join([
        '{"type":"thread.started","thread_id":"thread-123"}',
        '{"type":"turn.completed","turn_id":"turn-1"}',
    ])
    assert extract_thread_id(jsonl) == "thread-123"


def test_codex_resume_skips_output_flag():
    agent = CodexAgent()
    cmd = agent.build_command(
        prompt="Review this",
        output_file="/tmp/out.txt",
        session_id="thread-456",
    )
    assert cmd[:3] == ["codex", "exec", "resume"]
    assert "-o" not in cmd
    assert "/tmp/out.txt" not in cmd


def test_codex_non_resume_includes_output_flag():
    agent = CodexAgent()
    cmd = agent.build_command(
        prompt="Analyze this",
        output_file="/tmp/out.txt",
    )
    assert "exec" in cmd
    assert "resume" not in cmd
    assert "-o" in cmd
    assert "/tmp/out.txt" in cmd


def test_extract_message_content_from_completed():
    jsonl = "\n".join([
        '{"type":"thread.started","thread_id":"t-1"}',
        '{"type":"message.delta","content":"partial"}',
        '{"type":"message.completed","text":"{\\"error\\": false, \\"assessment\\": \\"Valid result\\"}"}',
    ])
    content = extract_message_content(jsonl)
    assert "error" in content
    assert "Valid result" in content


def test_extract_message_content_fallback_to_raw():
    raw = "not jsonl at all"
    assert extract_message_content(raw) == raw
