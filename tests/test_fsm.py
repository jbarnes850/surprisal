from surprisal.fsm import select_next_state, FSMResponse


def test_start_goes_to_generator():
    assert select_next_state("start", None, failure_count=0, revision_count=0) == "experiment_generator"


def test_generator_goes_to_runner():
    assert select_next_state("experiment_generator", FSMResponse(error=False), 0, 0) == "experiment_runner"


def test_runner_goes_to_analyst():
    assert select_next_state("experiment_runner", FSMResponse(error=False, exit_code=0), 0, 0) == "experiment_analyst"


def test_runner_infra_error_125():
    assert select_next_state("experiment_runner", FSMResponse(error=True, exit_code=125), 0, 0) == "FAIL"


def test_runner_infra_error_126():
    assert select_next_state("experiment_runner", FSMResponse(error=True, exit_code=126), 0, 0) == "FAIL"


def test_runner_infra_error_127():
    assert select_next_state("experiment_runner", FSMResponse(error=True, exit_code=127), 0, 0) == "FAIL"


def test_runner_code_error_goes_to_analyst():
    assert select_next_state("experiment_runner", FSMResponse(error=True, exit_code=1), 0, 0) == "experiment_analyst"


def test_analyst_error_retries_runner():
    assert select_next_state("experiment_analyst", FSMResponse(error=True), failure_count=2, revision_count=0) == "experiment_runner"


def test_analyst_error_at_max_fails():
    assert select_next_state("experiment_analyst", FSMResponse(error=True), failure_count=6, revision_count=0) == "FAIL"


def test_analyst_error_respects_custom_max_failures():
    assert select_next_state(
        "experiment_analyst",
        FSMResponse(error=True),
        failure_count=1,
        revision_count=0,
        max_failures=2,
    ) == "experiment_runner"
    assert select_next_state(
        "experiment_analyst",
        FSMResponse(error=True),
        failure_count=2,
        revision_count=0,
        max_failures=2,
    ) == "FAIL"


def test_analyst_success_goes_to_reviewer():
    assert select_next_state("experiment_analyst", FSMResponse(error=False), 0, 0) == "experiment_reviewer"


def test_reviewer_error_goes_to_reviser():
    assert select_next_state("experiment_reviewer", FSMResponse(error=True), 0, revision_count=0) == "experiment_reviser"


def test_reviewer_error_at_max_fails():
    assert select_next_state("experiment_reviewer", FSMResponse(error=True), 0, revision_count=1) == "FAIL"


def test_reviewer_error_respects_custom_max_revisions():
    assert select_next_state(
        "experiment_reviewer",
        FSMResponse(error=True),
        0,
        revision_count=0,
        max_revisions=0,
    ) == "FAIL"
    assert select_next_state(
        "experiment_reviewer",
        FSMResponse(error=True),
        0,
        revision_count=1,
        max_revisions=2,
    ) == "experiment_reviser"


def test_reviewer_success_goes_to_hypothesis_generator():
    assert select_next_state("experiment_reviewer", FSMResponse(error=False), 0, 0) == "hypothesis_generator"


def test_reviser_goes_to_runner():
    assert select_next_state("experiment_reviser", FSMResponse(error=False), 0, 0) == "experiment_runner"


def test_hypothesis_generator_goes_to_belief():
    assert select_next_state("hypothesis_generator", FSMResponse(error=False), 0, 0) == "belief_elicitation"


def test_belief_goes_to_complete():
    assert select_next_state("belief_elicitation", FSMResponse(error=False), 0, 0) == "COMPLETE"


def test_unknown_state_goes_to_fail():
    assert select_next_state("unknown_state", FSMResponse(error=False), 0, 0) == "FAIL"


def test_full_happy_path():
    states = []
    state = "start"
    while state not in ("COMPLETE", "FAIL"):
        state = select_next_state(state, FSMResponse(error=False, exit_code=0), 0, 0)
        states.append(state)
    assert states == [
        "experiment_generator", "experiment_runner",
        "experiment_analyst", "experiment_reviewer", "hypothesis_generator",
        "belief_elicitation", "COMPLETE",
    ]


def test_failure_count_cumulative():
    assert select_next_state("experiment_analyst", FSMResponse(error=True), failure_count=5, revision_count=0) == "experiment_runner"
    assert select_next_state("experiment_analyst", FSMResponse(error=True), failure_count=6, revision_count=0) == "FAIL"
