import asyncio
import tempfile
import time
from pathlib import Path
from typing import Optional
from .base import AgentResult


class CodexAgent:
    def __init__(self, model: str = "gpt-5.4"):
        self.model = model

    def build_command(self, prompt: str, output_file: Optional[str] = None) -> list[str]:
        cmd = ["codex", "exec", "--full-auto",
               "-c", f'model="{self.model}"']
        if output_file:
            cmd.extend(["-o", output_file])
        cmd.append(prompt)
        return cmd

    async def invoke(
        self, prompt: str, cwd: Optional[str] = None, timeout: int = 600
    ) -> AgentResult:
        # Use -o to capture the last message cleanly
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_file = f.name

        cmd = self.build_command(prompt, output_file=output_file)
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

        # Read the clean last-message output
        output_path = Path(output_file)
        if output_path.exists():
            raw = output_path.read_text()
            output_path.unlink(missing_ok=True)
        else:
            raw = stdout.decode("utf-8", errors="replace")

        return AgentResult.from_raw(
            raw, exit_code=proc.returncode or 0, duration=duration,
        )
