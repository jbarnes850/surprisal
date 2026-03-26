from dataclasses import dataclass
from typing import Optional


@dataclass
class FSMResponse:
    error: bool
    exit_code: Optional[int] = None
    feedback: Optional[str] = None
    data: Optional[dict] = None


INFRA_ERROR_CODES = {125, 126, 127}

STATES = [
    "start", "experiment_generator", "experiment_runner",
    "experiment_analyst", "experiment_reviewer",
    "experiment_reviser", "hypothesis_generator", "belief_elicitation",
]


def select_next_state(current_state: str, response: Optional[FSMResponse],
                       failure_count: int, revision_count: int) -> str:
    if current_state == "start":
        return "experiment_generator"
    elif current_state == "experiment_generator":
        return "experiment_runner"
    elif current_state == "experiment_runner":
        if response and response.exit_code in INFRA_ERROR_CODES:
            return "FAIL"
        return "experiment_analyst"
    elif current_state == "experiment_analyst":
        if response and response.error:
            if failure_count < 6:
                return "experiment_runner"
            return "FAIL"
        return "experiment_reviewer"
    elif current_state == "experiment_reviewer":
        if response and response.error:
            if revision_count < 1:
                return "experiment_reviser"
            return "FAIL"
        return "hypothesis_generator"
    elif current_state == "experiment_reviser":
        return "experiment_runner"
    elif current_state == "hypothesis_generator":
        return "belief_elicitation"
    elif current_state == "belief_elicitation":
        return "COMPLETE"
    return "FAIL"
