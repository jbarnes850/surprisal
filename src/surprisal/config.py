from dataclasses import dataclass, field, fields
from pathlib import Path
import tomllib


@dataclass
class GeneralConfig:
    default_budget: int = 100
    default_concurrency: int = 2


@dataclass
class MCTSConfig:
    c_explore: float = 1.414
    k_progressive: float = 1.0
    alpha_progressive: float = 0.5
    max_depth: int = 30
    belief_samples: int = 30
    belief_temperature: float = 0.7
    virtual_loss: int = 2
    dedup_interval: int = 50


@dataclass
class AgentsConfig:
    claude_model: str = "opus"
    codex_model: str = "gpt-5.4"
    max_turns: int = 20
    code_attempts: int = 6
    revision_attempts: int = 1
    generator_timeout: int = 180  # seconds, longer when literature search is active


@dataclass
class SandboxConfig:
    backend: str = "auto"
    image: str = "surprisal-gpu:latest"
    gpu: bool = True
    memory_limit: str = "16g"
    cpu_limit: str = "4"
    timeout: int = 1800
    network: bool = True
    hf_flavor: str = "a10g-small"
    hf_timeout: str = "2h"


@dataclass
class CredentialsConfig:
    wandb_api_key: str = ""
    hf_token: str = ""


@dataclass
class PredictorConfig:
    enabled: bool = False
    model_path: str = ""
    lambda_weight: float = 0.0
    min_training_samples: int = 100


@dataclass
class AutoDiscoveryConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    credentials: CredentialsConfig = field(default_factory=CredentialsConfig)

    def set(self, key: str, value: str) -> None:
        section_name, _, field_name = key.partition(".")
        if not field_name:
            raise KeyError(f"Invalid key: {key}")
        section = getattr(self, section_name, None)
        if section is None:
            raise KeyError(f"Unknown config section: {section_name}")
        if not hasattr(section, field_name):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(section, field_name)
        if isinstance(current, bool):
            setattr(section, field_name, value.lower() in ("true", "1", "yes"))
        elif isinstance(current, int):
            setattr(section, field_name, int(value))
        elif isinstance(current, float):
            setattr(section, field_name, float(value))
        else:
            setattr(section, field_name, value)


def load_config(path: Path) -> AutoDiscoveryConfig:
    cfg = AutoDiscoveryConfig()
    if not path.exists():
        return cfg
    with open(path, "rb") as f:
        data = tomllib.load(f)
    for section_name in ("general", "mcts", "agents", "sandbox", "predictor", "credentials"):
        if section_name in data:
            section = getattr(cfg, section_name)
            for k, v in data[section_name].items():
                if hasattr(section, k):
                    setattr(section, k, v)
    return cfg


def save_config(cfg: AutoDiscoveryConfig, path: Path) -> None:
    lines = []
    for section_name in ("general", "mcts", "agents", "sandbox", "predictor", "credentials"):
        section = getattr(cfg, section_name)
        lines.append(f"[{section_name}]")
        for f in fields(section):
            val = getattr(section, f.name)
            if isinstance(val, bool):
                lines.append(f"{f.name} = {'true' if val else 'false'}")
            elif isinstance(val, str):
                lines.append(f'{f.name} = "{val}"')
            else:
                lines.append(f"{f.name} = {val}")
        lines.append("")
    path.write_text("\n".join(lines))
