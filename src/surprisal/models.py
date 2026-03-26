from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Node:
    id: str
    exploration_id: str
    hypothesis: str
    parent_id: Optional[str] = None
    initial_hypothesis: Optional[str] = None
    context: Optional[str] = None
    variables: Optional[str] = None  # JSON array
    relationships: Optional[str] = None  # JSON array
    cited_papers: Optional[str] = None  # JSON array of {arxiv_id, title, gap}
    depth: int = 0
    visit_count: int = 0
    virtual_loss: int = 0
    surprisal_sum: float = 0.0
    bayesian_surprise: Optional[float] = None
    belief_shifted: Optional[bool] = None
    prior_alpha: Optional[float] = None
    prior_beta: Optional[float] = None
    posterior_alpha: Optional[float] = None
    posterior_beta: Optional[float] = None
    k_prior: Optional[int] = None
    k_post: Optional[int] = None
    n_belief_samples: int = 30
    status: str = "pending"
    branch_id: Optional[str] = None
    claude_session_id: Optional[str] = None
    codex_session_id: Optional[str] = None
    experiment_exit_code: Optional[int] = None
    fsm_state: str = "start"
    fsm_failure_count: int = 0
    fsm_revision_count: int = 0
    dedup_cluster_id: Optional[str] = None
    created_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None


@dataclass
class BeliefSample:
    node_id: str
    phase: str  # 'prior' or 'posterior'
    sample_index: int
    believes_hypothesis: bool
    raw_response: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class AgentInvocation:
    node_id: str
    role: str
    provider: str  # 'claude', 'codex', 'docker'
    prompt_hash: Optional[str] = None
    response_hash: Optional[str] = None
    duration_seconds: Optional[float] = None
    exit_code: Optional[int] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class Exploration:
    id: str
    domain: str
    dataset_path: Optional[str] = None
    status: str = "initialized"
    budget: int = 100
    created_at: Optional[datetime] = None
