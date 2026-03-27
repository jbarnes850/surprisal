import pytest
from unittest.mock import patch, AsyncMock
from surprisal.agents.backends import detect_gpu, create_backend
from surprisal.config import DEFAULT_CPU_IMAGE, DEFAULT_GPU_IMAGE, SandboxConfig, CredentialsConfig


@pytest.mark.asyncio
async def test_detect_gpu_with_nvidia_smi():
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"GPU 0: NVIDIA GB10\n", b"")
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc
        result = await detect_gpu()
        assert result is True


@pytest.mark.asyncio
async def test_detect_gpu_without_nvidia_smi():
    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
        result = await detect_gpu()
        assert result is False


def test_create_backend_local():
    from surprisal.agents.experiment_container import ExperimentContainer
    config = SandboxConfig(backend="local")
    creds = CredentialsConfig()
    backend = create_backend(config, creds)
    assert isinstance(backend, ExperimentContainer)


def test_create_backend_hf_jobs():
    from surprisal.agents.hf_jobs import HFJobsSandbox
    config = SandboxConfig(backend="hf_jobs")
    creds = CredentialsConfig(hf_token="test")
    backend = create_backend(config, creds)
    assert isinstance(backend, HFJobsSandbox)


def test_create_backend_auto_returns_host_runner():
    from surprisal.agents.backends import HostRunner
    config = SandboxConfig(backend="auto")
    creds = CredentialsConfig()
    backend = create_backend(config, creds, gpu_available=True)
    assert isinstance(backend, HostRunner)


def test_create_backend_docker_with_gpu():
    from surprisal.agents.experiment_container import ExperimentContainer
    config = SandboxConfig(backend="docker")
    creds = CredentialsConfig()
    backend = create_backend(config, creds, gpu_available=True)
    assert isinstance(backend, ExperimentContainer)
    assert backend.config.gpu is True
    assert backend.config.image == DEFAULT_GPU_IMAGE


def test_create_backend_docker_without_gpu():
    from surprisal.agents.experiment_container import ExperimentContainer
    config = SandboxConfig(backend="docker")
    creds = CredentialsConfig()
    backend = create_backend(config, creds, gpu_available=False)
    assert isinstance(backend, ExperimentContainer)
    assert backend.config.gpu is False
    assert backend.config.image == DEFAULT_CPU_IMAGE
