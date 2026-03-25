import asyncio
import time
from pathlib import Path
from typing import Optional
from .base import AgentResult


INFRA_ERROR_CODES = {125, 126, 127}


class DockerSandbox:
    def __init__(
        self,
        image: str = "autodiscovery-sandbox:latest",
        memory: str = "2g",
        cpus: str = "1.5",
        timeout: int = 600,
        network: bool = False,
    ):
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.timeout = timeout
        self.network = network

    def build_run_command(self, experiment_dir: str) -> list[str]:
        cmd = ["docker", "run", "--rm"]
        if not self.network:
            cmd.append("--network=none")
        cmd.extend(["--memory", self.memory])
        cmd.extend(["--cpus", self.cpus])
        cmd.extend(["--pids-limit", "256"])
        cmd.append("--read-only")
        cmd.extend(["--tmpfs", "/tmp:rw,size=512m"])
        cmd.extend(["-v", f"{experiment_dir}:/work:rw"])
        cmd.append(self.image)
        cmd.extend(["timeout", str(self.timeout), "python", "/work/experiment.py"])
        return cmd

    @staticmethod
    def is_infra_error(exit_code: int) -> bool:
        return exit_code in INFRA_ERROR_CODES

    async def execute(self, experiment_dir: str) -> AgentResult:
        cmd = self.build_run_command(experiment_dir)
        stdout_path = Path(experiment_dir) / "stdout.txt"
        stderr_path = Path(experiment_dir) / "stderr.txt"
        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout + 30
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return AgentResult(raw="", exit_code=-1, duration_seconds=self.timeout)
        duration = time.monotonic() - start
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        return AgentResult.from_raw(
            stdout.decode("utf-8", errors="replace"),
            exit_code=proc.returncode or 0,
            duration=duration,
        )

    @staticmethod
    async def build_image(dockerfile_dir: str, tag: str = "autodiscovery-sandbox:latest"):
        proc = await asyncio.create_subprocess_exec(
            "docker", "build", "-t", tag, dockerfile_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0
