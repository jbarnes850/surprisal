import asyncio
import json
import time
from typing import Optional

from .base import AgentResult


class ClaudeAgent:
    def __init__(self, model: str = "opus", max_turns: int = 20):
        self.model = model
        self.max_turns = max_turns

    def build_command(
        self,
        prompt: str,
        system_prompt_file: Optional[str] = None,
        output_format: str = "json",
        json_schema: Optional[dict] = None,
        cwd: Optional[str] = None,
        no_tools: bool = False,
        extra_args: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        resume_session: bool = False,
        fork_session: bool = False,
    ) -> list[str]:
        cmd = ["claude", "-p", prompt]
        cmd.extend(["--output-format", output_format])
        cmd.extend(["--model", self.model])
        cmd.extend(["--max-turns", str(self.max_turns)])
        cmd.extend([
            "--dangerously-skip-permissions",
            "--disable-slash-commands",
            "--setting-sources",
            "",
        ])
        if no_tools:
            cmd.extend(["--disallowedTools", "Bash", "Edit", "Write", "Read",
                        "Glob", "Grep", "WebSearch", "WebFetch"])
        if resume_session and session_id:
            cmd.extend(["--resume", session_id])
        elif session_id:
            cmd.extend(["--session-id", session_id])
        if fork_session:
            cmd.append("--fork-session")
        if system_prompt_file:
            cmd.extend(["--system-prompt-file", system_prompt_file])
        if json_schema:
            cmd.extend(["--json-schema", json.dumps(json_schema)])
        if extra_args:
            cmd.extend(extra_args)
        return cmd

    async def invoke(
        self,
        prompt: str,
        system_prompt_file: Optional[str] = None,
        output_format: str = "json",
        json_schema: Optional[dict] = None,
        cwd: Optional[str] = None,
        timeout: int = 600,
        no_tools: bool = False,
        extra_args: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        resume_session: bool = False,
        fork_session: bool = False,
    ) -> AgentResult:
        cmd = self.build_command(
            prompt=prompt,
            system_prompt_file=system_prompt_file,
            output_format="json",
            json_schema=json_schema,
            no_tools=no_tools,
            extra_args=extra_args,
            session_id=session_id,
            resume_session=resume_session,
            fork_session=fork_session,
        )
        import os
        env = os.environ.copy()
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
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
        raw_stdout = stdout.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw_stdout)
        except json.JSONDecodeError:
            parsed = None

        if output_format == "json" or not isinstance(parsed, dict):
            return AgentResult.from_raw(
                raw_stdout,
                exit_code=proc.returncode or 0,
                duration=duration,
                parsed=parsed,
            )

        payload = parsed.get("result", raw_stdout)
        if isinstance(payload, str):
            raw = payload
        else:
            raw = json.dumps(payload)
        return AgentResult.from_raw(
            raw,
            exit_code=proc.returncode or 0,
            duration=duration,
            parsed=parsed,
        )
