"""Runtime MCP config generation with credential injection."""
import json
import tempfile

from surprisal.config import CredentialsConfig


def generate_mcp_config(credentials: CredentialsConfig) -> dict:
    config: dict = {"mcpServers": {}}
    if credentials.wandb_api_key:
        config["mcpServers"]["wandb"] = {
            "command": "npx",
            "args": ["-y", "@wandb/mcp-server"],
            "env": {"WANDB_API_KEY": credentials.wandb_api_key},
        }
    if credentials.hf_token:
        config["mcpServers"]["huggingface"] = {
            "command": "npx",
            "args": ["-y", "@huggingface/mcp-server"],
            "env": {"HF_TOKEN": credentials.hf_token},
        }
    return config


def write_mcp_config(credentials: CredentialsConfig) -> str:
    config = generate_mcp_config(credentials)
    fd, path = tempfile.mkstemp(suffix=".json", prefix="surprisal-mcp-")
    with open(fd, "w") as f:
        json.dump(config, f)
    return path
