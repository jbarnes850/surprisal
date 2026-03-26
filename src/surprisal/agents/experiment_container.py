"""Local container backend — runs the runner agent inside Docker."""
import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from surprisal.agents.base import AgentResult
from surprisal.config import (
    CredentialsConfig,
    DEFAULT_CPU_IMAGE,
    DEFAULT_GPU_IMAGE,
    SandboxConfig,
    resolve_sandbox_image,
)
from surprisal.mcp_config import write_mcp_config
from surprisal.progress import ProgressCallback, emit_progress


INFRA_ERROR_CODES = {125, 126, 127}
RUNNER_CLAUDE_STATE_DIR = ".claude-runner"
RUNNER_CLAUDE_SEED_FILES = (".credentials.json", "settings.json", "settings.local.json")
RUNNER_CLAUDE_HOME_FILE = ".claude.json"
CONTAINER_HOME = "/home/surprisal"
RUNNER_AUTH_ENV_VARS = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
)


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
            cmd.extend(["-v", f"{claude_home}:{CONTAINER_HOME}:rw"])
            cmd.extend(["-e", f"HOME={CONTAINER_HOME}"])
        for env_name in RUNNER_AUTH_ENV_VARS:
            env_value = os.environ.get(env_name)
            if env_value:
                cmd.extend(["-e", f"{env_name}={env_value}"])
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
        host_claude_json = Path.home() / RUNNER_CLAUDE_HOME_FILE
        if not host_claude_home.exists():
            return None

        branch_workspace = workspace.parent.parent
        runner_claude_home = branch_workspace / RUNNER_CLAUDE_STATE_DIR
        runner_claude_home.mkdir(parents=True, exist_ok=True)
        runner_claude_dir = runner_claude_home / ".claude"
        runner_claude_dir.mkdir(parents=True, exist_ok=True)

        for relative_name in RUNNER_CLAUDE_SEED_FILES:
            source = host_claude_home / relative_name
            destination = runner_claude_dir / relative_name
            if source.exists() and not destination.exists():
                shutil.copy2(source, destination)

        destination_json = runner_claude_home / RUNNER_CLAUDE_HOME_FILE
        if host_claude_json.exists() and not destination_json.exists():
            shutil.copy2(host_claude_json, destination_json)

        return runner_claude_home

    @staticmethod
    def is_infra_error(exit_code: int) -> bool:
        return exit_code in INFRA_ERROR_CODES

    @staticmethod
    async def image_exists(tag: str) -> bool:
        proc = await asyncio.create_subprocess_exec(
            "docker", "image", "inspect", tag,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        return proc.returncode == 0

    @staticmethod
    def dockerfile_for_tag(tag: str) -> Path | None:
        sandbox_dir = Path(__file__).resolve().parents[3] / "sandbox"
        mapping = {
            DEFAULT_CPU_IMAGE: sandbox_dir / "Dockerfile.cpu",
            DEFAULT_GPU_IMAGE: sandbox_dir / "Dockerfile.gpu",
        }
        return mapping.get(tag)

    @classmethod
    async def ensure_image(
        cls,
        tag: str,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[bool, str]:
        if await cls.image_exists(tag):
            emit_progress(progress_callback, f"Runner: using existing local image `{tag}`.")
            return True, ""

        dockerfile_path = cls.dockerfile_for_tag(tag)
        if dockerfile_path is None:
            return False, f"docker image '{tag}' is missing and has no known build recipe"
        if not dockerfile_path.exists():
            return False, f"docker image '{tag}' is missing and {dockerfile_path.name} was not found"

        if tag == DEFAULT_CPU_IMAGE:
            emit_progress(progress_callback, "CPU image missing, building once for local execution.")
        elif tag == DEFAULT_GPU_IMAGE:
            emit_progress(progress_callback, "GPU image missing, building once for local execution.")
        else:
            emit_progress(progress_callback, f"Sandbox image `{tag}` is missing, building it now.")
        emit_progress(progress_callback, f"Runner: building `{tag}` from `{dockerfile_path.name}`.")
        ok = await cls.build_image(str(dockerfile_path), tag=tag)
        if not ok:
            return False, f"failed to build docker image '{tag}' from {dockerfile_path.name}"
        emit_progress(progress_callback, f"Runner: finished building `{tag}`.")
        return True, ""

    async def execute(
        self,
        experiment_prompt: str,
        workspace: Path,
        config: SandboxConfig,
        system_prompt_file: str | None = None,
        session_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AgentResult:
        mcp_config_path = write_mcp_config(self.credentials)
        try:
            self.config.image = resolve_sandbox_image(self.config.image, self.config.gpu)
            emit_progress(progress_callback, f"Runner: preparing local sandbox image `{self.config.image}`.")
            image_ok, image_error = await self.ensure_image(
                self.config.image,
                progress_callback=progress_callback,
            )
            if not image_ok:
                return AgentResult(raw=image_error, exit_code=1)
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
            emit_progress(progress_callback, f"Runner: launching local sandbox container `{self.config.image}`.")
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
            emit_progress(progress_callback, f"Runner: container finished with exit code {proc.returncode or 0}.")
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
            "docker", "build", "--progress=plain", "-t", tag, "-f", dockerfile_path, dockerfile_dir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0
