import json
from surprisal.mcp_config import generate_mcp_config
from surprisal.config import CredentialsConfig


def test_empty_credentials_produces_empty_servers():
    creds = CredentialsConfig()
    config = generate_mcp_config(creds)
    assert config == {"mcpServers": {}}


def test_wandb_only():
    creds = CredentialsConfig(wandb_api_key="test-key")
    config = generate_mcp_config(creds)
    assert "wandb" in config["mcpServers"]
    assert config["mcpServers"]["wandb"]["env"]["WANDB_API_KEY"] == "test-key"
    assert "huggingface" not in config["mcpServers"]


def test_hf_only():
    creds = CredentialsConfig(hf_token="hf_test")
    config = generate_mcp_config(creds)
    assert "huggingface" in config["mcpServers"]
    assert config["mcpServers"]["huggingface"]["env"]["HF_TOKEN"] == "hf_test"
    assert "wandb" not in config["mcpServers"]


def test_both_credentials():
    creds = CredentialsConfig(wandb_api_key="wk", hf_token="hf")
    config = generate_mcp_config(creds)
    assert "wandb" in config["mcpServers"]
    assert "huggingface" in config["mcpServers"]


def test_output_is_json_serializable():
    creds = CredentialsConfig(wandb_api_key="k", hf_token="t")
    config = generate_mcp_config(creds)
    serialized = json.dumps(config)
    assert json.loads(serialized) == config
