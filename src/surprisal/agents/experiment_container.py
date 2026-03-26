"""Local GPU container backend — runs agent inside Docker container."""
import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from surprisal.agents.base import AgentResult
from surprisal.config import SandboxConfig, CredentialsConfig
from surprisal.mcp_config import write_mcp_config


INFRA_ERROR_CODES = {125, 126, 127}
RUNNER_CLAUDE_STATE_DIR = ".claude-runner"
RUNNER_CLAUDE_SEED_FILES = (".credentials.json", "settings.json", "settings.local.json")


class ExperimentContainer:
    def __init__(self, config: SandboxConfig, credentials: CredentialsConfig):
        self.config = config
        self.credentials = credentials

    def build_run_command(
        self,
        workspace: str,
        mcp_config: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        claude_home: Optional[str] = None,
    ) -> list[str]:
        cmd = ["docker", "run", "--rm"]
        if self.config.gpu:
            cmd.append("--gpus=all")
        cmd.extend(["--memory", self.config.memory_limit])
        cmd.extend(["--cpus", self.config.cpu_limit])
        if not self.config.network:
            cmd.extend(["--network", "none"])
        cmd.extend(["-v", f"{workspace}:/work:rw"])
        cmd.extend(["-v", f"{mcp_config}:/etc/surprisal/mcp.json:ro"])
        if claude_home:
            cmd.extend(["-v", f"{claude_home}:/root/.claude:rw"])
            cmd.extend(["-e", "HOME=/root"])
        if self.credentials.wandb_api_key:
            cmd.extend(["-e", f"WANDB_API_KEY={self.credentials.wandb_api_key}"])
        if self.credentials.hf_token:
            cmd.extend(["-e", f"HF_TOKEN={self.credentials.hf_token}"])
        cmd.append(self.config.image)
        cmd.extend([
            "claude", "-p", prompt,
            "--mcp-config", "/etc/surprisal/mcp.json",
            "--dangerously-skip-permissions",
            "--output-format", "json",
            "--setting-sources", "",
        ])
        if session_id:
            cmd.extend(["--resume", session_id])
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])
        return cmd

    @staticmethod
    def prepare_runner_claude_home(workspace: Path) -> Path | None:
        host_claude_home = Path.home() / ".claude"
        if not host_claude_home.exists():
            return None

        branch_workspace = workspace.parent.parent
        runner_claude_home = branch_workspace / RUNNER_CLAUDE_STATE_DIR
        runner_claude_home.mkdir(parents=True, exist_ok=True)

        for relative_name in RUNNER_CLAUDE_SEED_FILES:
            source = host_claude_home / relative_name
            destination = runner_claude_home / relative_name
            if source.exists() and not destination.exists():
                shutil.copy2(source, destination)

        return runner_claude_home

    @staticmethod
    def is_infra_error(exit_code: int) -> bool:
        return exit_code in INFRA_ERROR_CODES

    async def execute(
        self,
        experiment_prompt: str,
        workspace: Path,
        config: SandboxConfig,
        system_prompt_file: str | None = None,
        session_id: str | None = None,
    ) -> AgentResult:
        mcp_config_path = write_mcp_config(self.credentials)
        try:
            system_prompt = Path(system_prompt_file).read_text() if system_prompt_file else None
            claude_home = self.prepare_runner_claude_home(workspace)
            if session_id and claude_home is None:
                return AgentResult(
                    raw="runner session persistence unavailable: host Claude home not found",
                    exit_code=1,
                )
            cmd = self.build_run_command(
                str(workspace),
                mcp_config_path,
                experiment_prompt,
                system_prompt=system_prompt,
                session_id=session_id,
                claude_home=str(claude_home) if claude_home is not None else None,
            )
            start = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout + 30)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return AgentResult(raw="", exit_code=-1, duration_seconds=self.config.timeout)
            duration = time.monotonic() - start
            stdout_path = workspace / "stdout.txt"
            stderr_path = workspace / "stderr.txt"
            stdout_path.write_bytes(stdout)
            stderr_path.write_bytes(stderr)
            return AgentResult.from_raw(
                stdout.decode("utf-8", errors="replace"),
                exit_code=proc.returncode or 0,
                duration=duration,
            )
        finally:
            try:
                os.unlink(mcp_config_path)
            except OSError:
                pass

    @staticmethod
    async def build_image(dockerfile_path: str, tag: str = "surprisal-gpu:latest"):
        dockerfile_dir = str(Path(dockerfile_path).parent)
        proc = await asyncio.create_subprocess_exec(
            "docker", "build", "-t", tag, "-f", dockerfile_path, dockerfile_dir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0
