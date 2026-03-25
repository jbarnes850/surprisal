import asyncio
import json
import time
from pathlib import Path
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
        session_id: Optional[str] = None,
        resume_session: Optional[str] = None,
        fork_session: bool = False,
        json_schema: Optional[dict] = None,
        cwd: Optional[str] = None,
        no_tools: bool = False,
    ) -> list[str]:
        cmd = ["claude", "-p", prompt]
        cmd.extend(["--output-format", output_format])
        cmd.extend(["--model", self.model])
        cmd.extend(["--max-turns", str(self.max_turns)])
        cmd.extend(["--dangerously-skip-permissions", "--no-session-persistence"])
        if no_tools:
            cmd.extend(["--disallowedTools", "Bash", "Edit", "Write", "Read",
                        "Glob", "Grep", "WebSearch", "WebFetch"])
        if system_prompt_file:
            cmd.extend(["--system-prompt-file", system_prompt_file])
        if resume_session:
            cmd.extend(["--resume", resume_session])
            if fork_session:
                cmd.append("--fork-session")
        elif session_id:
            cmd.extend(["--session-id", session_id])
        if json_schema:
            cmd.extend(["--json-schema", json.dumps(json_schema)])
        return cmd

    async def invoke(
        self,
        prompt: str,
        system_prompt_file: Optional[str] = None,
        output_format: str = "json",
        session_id: Optional[str] = None,
        resume_session: Optional[str] = None,
        fork_session: bool = False,
        json_schema: Optional[dict] = None,
        cwd: Optional[str] = None,
        timeout: int = 600,
        no_tools: bool = False,
    ) -> AgentResult:
        cmd = self.build_command(
            prompt=prompt,
            system_prompt_file=system_prompt_file,
            output_format=output_format,
            session_id=session_id,
            resume_session=resume_session,
            fork_session=fork_session,
            json_schema=json_schema,
            no_tools=no_tools,
        )
        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return AgentResult(raw="", exit_code=-1, duration_seconds=timeout)
        duration = time.monotonic() - start
        return AgentResult.from_raw(
            stdout.decode("utf-8", errors="replace"),
            exit_code=proc.returncode or 0,
            duration=duration,
        )
