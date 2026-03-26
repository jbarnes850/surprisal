"""HF Jobs cloud backend — submits experiments to HuggingFace Jobs."""
import asyncio
import logging
from pathlib import Path

from surprisal.agents.base import AgentResult
from surprisal.agents.claude import ClaudeAgent
from surprisal.config import SandboxConfig, CredentialsConfig

logger = logging.getLogger("surprisal")


def extract_code(text: str) -> str:
    """Extract Python code from text that may contain markdown fences."""
    stripped = text.strip()
    if "```python" in stripped:
        start = stripped.index("```python") + len("```python")
        end = stripped.index("```", start) if "```" in stripped[start:] else len(stripped)
        return stripped[start:end].strip()
    if stripped.startswith("```"):
        stripped = stripped[3:].strip()
    if stripped.endswith("```"):
        stripped = stripped[:-3].strip()
    return stripped


class HFJobsSandbox:
    def __init__(self, config: SandboxConfig, credentials: CredentialsConfig):
        self.config = config
        self.credentials = credentials

    async def execute(
        self,
        experiment_prompt: str,
        workspace: Path,
        config: SandboxConfig,
        system_prompt_file: str | None = None,
        session_id: str | None = None,
    ) -> AgentResult:
        model = "opus"
        agent = ClaudeAgent(model=model, max_turns=3)
        script_result = await agent.invoke(
            prompt=(
                f"Write a self-contained Python script for:\n{experiment_prompt}\n\n"
                "Print results as JSON to stdout. Include all imports.\n"
                "Output ONLY code, no explanation."
            ),
            system_prompt_file=system_prompt_file,
            output_format="text",
            no_tools=True,
            session_id=session_id,
            resume_session=bool(session_id),
        )
        script = extract_code(script_result.raw)
        script_path = workspace / "experiment.py"
        script_path.write_text(script)

        try:
            from huggingface_hub import run_uv_job, fetch_job_logs, inspect_job
        except ImportError:
            logger.error("huggingface_hub not installed — cannot use HF Jobs backend")
            return AgentResult(raw="huggingface_hub not installed", exit_code=1)

        secrets = {}
        if self.credentials.wandb_api_key:
            secrets["WANDB_API_KEY"] = self.credentials.wandb_api_key
        if self.credentials.hf_token:
            secrets["HF_TOKEN"] = self.credentials.hf_token

        job = run_uv_job(
            str(script_path),
            flavor=self.config.hf_flavor,
            dependencies=["torch", "transformers", "wandb", "datasets", "numpy", "scipy", "pandas", "scikit-learn"],
            secrets=secrets,
            timeout=self.config.hf_timeout,
        )
        logger.info(f"HF Job submitted: {job.id}")

        while True:
            info = inspect_job(job_id=job.id)
            if info.status.stage in ("COMPLETED", "ERROR"):
                break
            await asyncio.sleep(10)

        logs = list(fetch_job_logs(job_id=job.id))
        raw = "\n".join(str(log) for log in logs)
        exit_code = 0 if info.status.stage == "COMPLETED" else 1
        logger.info(f"HF Job {job.id} finished: {info.status.stage}")
        return AgentResult.from_raw(
            raw,
            exit_code=exit_code,
            session_id=script_result.session_id,
        )
