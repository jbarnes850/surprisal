from surprisal.agents.hf_jobs import HFJobsSandbox, extract_code
from surprisal.config import SandboxConfig, CredentialsConfig


def test_extract_code_from_markdown():
    text = "Here is the code:\n```python\nimport numpy as np\nprint(42)\n```\nDone."
    assert extract_code(text) == "import numpy as np\nprint(42)"


def test_extract_code_plain():
    text = "import numpy as np\nprint(42)"
    assert extract_code(text) == "import numpy as np\nprint(42)"


def test_extract_code_triple_backtick():
    text = "```\nimport os\n```"
    assert extract_code(text) == "import os"


def test_hf_jobs_sandbox_init():
    config = SandboxConfig(hf_flavor="t4-small", hf_timeout="1h")
    creds = CredentialsConfig(hf_token="hf_test")
    sandbox = HFJobsSandbox(config=config, credentials=creds)
    assert sandbox.config.hf_flavor == "t4-small"
    assert sandbox.credentials.hf_token == "hf_test"
