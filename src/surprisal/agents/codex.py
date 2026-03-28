import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Optional

from .base import AgentResult


def extract_thread_id(jsonl_output: str) -> Optional[str]:
    """Extract the persisted thread ID from Codex JSONL output."""
    for line in jsonl_output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started":
            thread_id = event.get("thread_id")
            if isinstance(thread_id, str) and thread_id.strip():
                return thread_id
    return None


def extract_message_content(jsonl_output: str) -> str:
    """Extract final assistant message content from Codex JSONL output.

    Needed when -o (output-last-message) is unavailable (e.g. exec resume).
    Scans for message events and returns the last assistant content found.
    Falls back to the raw JSONL if no message content is found.
    """
    last_content = ""
    for line in jsonl_output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = event.get("type", "")
        # message.completed carries the full final text
        if etype == "message.completed":
            text = event.get("text") or event.get("content") or ""
            if isinstance(text, str) and text.strip():
                last_content = text.strip()
        # Accumulate deltas as fallback
        elif etype == "message.delta":
            delta = event.get("delta") or event.get("content") or ""
            if isinstance(delta, str):
                last_content += delta
    return last_content if last_content else jsonl_output


class CodexAgent:
    def __init__(self, model: str = "gpt-5.4"):
        self.model = model

    def compose_prompt(self, prompt: str, system_prompt_file: Optional[str] = None) -> str:
        if not system_prompt_file:
            return prompt
        system_prompt = Path(system_prompt_file).read_text().strip()
        return f"{system_prompt}\n\nUser task:\n{prompt}"

    def build_command(
        self,
        prompt: str,
        output_file: Optional[str] = None,
        *,
        output_schema_file: Optional[str] = None,
        extra_args: Optional[list[str]] = None,
        session_id: Optional[str] = None,
    ) -> list[str]:
        if session_id:
            cmd = ["codex", "exec", "resume", "--full-auto", "-c", f'model="{self.model}"']
        else:
            cmd = ["codex", "exec", "--full-auto", "-c", f'model="{self.model}"']
        cmd.append("--json")
        cmd.append("--skip-git-repo-check")
        # -o (output-last-message) is only supported by `exec`, not `exec resume`
        if output_file and not session_id:
            cmd.extend(["-o", output_file])
        if output_schema_file:
            cmd.extend(["--output-schema", output_schema_file])
        if extra_args:
            cmd.extend(extra_args)
        if session_id:
            cmd.append(session_id)
        cmd.append(prompt)
        return cmd

    async def invoke(
        self, prompt: str, cwd: Optional[str] = None, timeout: int = 600,
        output_format: str = "text", system_prompt_file: Optional[str] = None,
        no_tools: bool = False, extra_args: list[str] = None,
        json_schema: Optional[dict] = None,
        session_id: Optional[str] = None,
        resume_session: bool = False,
        **kwargs,
    ) -> AgentResult:
        if no_tools:
            raise ValueError("CodexAgent does not support no_tools=True")

        # Use -o to capture the last message cleanly
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_file = f.name

        schema_path = None
        if json_schema is not None:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                schema_path = f.name
                json.dump(json_schema, f)

        cmd = self.build_command(
            self.compose_prompt(prompt, system_prompt_file=system_prompt_file),
            output_file=output_file,
            output_schema_file=schema_path,
            extra_args=extra_args,
            session_id=session_id if resume_session and session_id else None,
        )
        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return AgentResult(
                raw="",
                exit_code=-1,
                duration_seconds=timeout,
                session_id=session_id if resume_session else None,
            )
        duration = time.monotonic() - start
        stdout_text = stdout.decode("utf-8", errors="replace")
        thread_id = extract_thread_id(stdout_text) or (session_id if resume_session else None)

        # Read the clean last-message output.
        # When -o is available (non-resume), the output file has the last message.
        # When -o is unavailable (resume), extract from JSONL stdout.
        output_path = Path(output_file)
        raw = ""
        if output_path.exists():
            raw = output_path.read_text()
            output_path.unlink(missing_ok=True)
        if not raw.strip():
            raw = extract_message_content(stdout_text)
        if schema_path:
            Path(schema_path).unlink(missing_ok=True)

        return AgentResult.from_raw(
            raw,
            exit_code=proc.returncode or 0,
            duration=duration,
            session_id=thread_id,
        )
