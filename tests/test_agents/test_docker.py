from surprisal.agents.docker import DockerSandbox


def test_docker_builds_run_command():
    sandbox = DockerSandbox(memory="2g", cpus="1.5", timeout=600, network=False)
    cmd = sandbox.build_run_command("/tmp/exp")
    assert "docker" in cmd
    assert "--network=none" in cmd
    assert "--memory" in cmd
    assert "2g" in cmd
    assert "--cpus" in cmd
    assert "--pids-limit" in cmd
    assert "256" in cmd
    assert "--read-only" in cmd
    assert "--tmpfs" in cmd
    assert "-v" in cmd
    assert "/tmp/exp:/work:rw" in cmd
    assert "timeout" in cmd
    assert "python" in cmd
    assert "/work/experiment.py" in cmd


def test_docker_network_enabled():
    sandbox = DockerSandbox(network=True)
    cmd = sandbox.build_run_command("/tmp/exp")
    assert "--network=none" not in cmd


def test_is_infra_error():
    assert DockerSandbox.is_infra_error(125) is True
    assert DockerSandbox.is_infra_error(126) is True
    assert DockerSandbox.is_infra_error(127) is True
    assert DockerSandbox.is_infra_error(1) is False
    assert DockerSandbox.is_infra_error(0) is False
    assert DockerSandbox.is_infra_error(137) is False  # OOM kill, not infra
