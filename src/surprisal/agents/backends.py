"""Sandbox backend protocol, auto-detection, and factory."""
import asyncio
from pathlib import Path
from typing import Protocol, runtime_checkable

from surprisal.agents.base import AgentResult
from surprisal.config import CredentialsConfig, SandboxConfig, resolve_sandbox_image
from surprisal.progress import ProgressCallback, emit_progress


@runtime_checkable
class SandboxBackend(Protocol):
    async def execute(
        self,
        experiment_prompt: str,
        workspace: Path,
        config: SandboxConfig,
        system_prompt_file: str | None = None,
        session_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
        node_id: str | None = None,
    ) -> AgentResult:
        ...


class HostRunner:
    """Host-native backend — runs the runner agent directly via ClaudeAgent."""

    async def execute(
        self,
        experiment_prompt: str,
        workspace: Path,
        config: SandboxConfig,
        system_prompt_file: str | None = None,
        session_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
        node_id: str | None = None,
    ) -> AgentResult:
        from surprisal.agents.claude import ClaudeAgent

        agent = ClaudeAgent(model="sonnet", max_turns=20)
        emit_progress(progress_callback, f"Runner: executing experiment on host.")
        result = await agent.invoke(
            prompt=experiment_prompt,
            system_prompt_file=system_prompt_file,
            output_format="json",
            cwd=str(workspace),
            timeout=config.timeout,
            session_id=session_id,
            resume_session=bool(session_id),
        )
        # If session resume failed instantly, retry with a fresh session.
        # Stale sessions (from killed runs) cause exit code 1 at 0s.
        if result.exit_code != 0 and session_id and result.duration_seconds < 2:
            emit_progress(progress_callback, f"Runner: session resume failed, retrying fresh.")
            result = await agent.invoke(
                prompt=experiment_prompt,
                system_prompt_file=system_prompt_file,
                output_format="json",
                cwd=str(workspace),
                timeout=config.timeout,
            )
        # Write stdout for downstream consumers (analyst, reviewer)
        (workspace / "stdout.txt").write_text(result.raw)
        return result


async def detect_gpu() -> bool:
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        return proc.returncode == 0 and len(stdout) > 0
    except (FileNotFoundError, asyncio.TimeoutError):
        return False


def create_backend(
    config: SandboxConfig,
    credentials: CredentialsConfig,
    gpu_available: bool | None = None,
) -> SandboxBackend:
    from surprisal.agents.experiment_container import ExperimentContainer
    from surprisal.agents.hf_jobs import HFJobsSandbox

    if config.backend == "hf_jobs":
        return HFJobsSandbox(config=config, credentials=credentials)

    if config.backend == "docker" or config.backend == "local":
        if gpu_available is not None:
            config.gpu = gpu_available
        config.image = resolve_sandbox_image(config.image, config.gpu)
        return ExperimentContainer(config=config, credentials=credentials)

    # backend == "auto" or anything else: host-native runner (no Docker)
    return HostRunner()
