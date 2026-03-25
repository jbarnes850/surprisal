import asyncio
import time
from typing import Optional
from .base import AgentResult


class CodexAgent:
    def __init__(self, model: str = "gpt-5.4"):
        self.model = model

    def build_command(self, prompt: str, cwd: Optional[str] = None) -> list[str]:
        cmd = ["codex", "-q", "--full-auto", "--model", self.model]
        if cwd:
            cmd.extend(["-f", cwd])
        cmd.append(prompt)
        return cmd

    async def invoke(
        self, prompt: str, cwd: Optional[str] = None, timeout: int = 600
    ) -> AgentResult:
        cmd = self.build_command(prompt, cwd)
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
            return AgentResult(raw="", exit_code=-1, duration_seconds=timeout)
        duration = time.monotonic() - start
        return AgentResult.from_raw(
            stdout.decode("utf-8", errors="replace"),
            exit_code=proc.returncode or 0,
            duration=duration,
        )
