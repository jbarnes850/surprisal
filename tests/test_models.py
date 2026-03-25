import json
from surprisal.models import Node, BeliefSample, AgentInvocation, Exploration


def test_node_creation_with_defaults():
    node = Node(id="abc", exploration_id="exp1", hypothesis="test hypothesis")
    assert node.depth == 0
    assert node.visit_count == 0
    assert node.virtual_loss == 0
    assert node.surprisal_sum == 0.0
    assert node.status == "pending"
    assert node.fsm_state == "start"
    assert node.fsm_failure_count == 0
    assert node.fsm_revision_count == 0
    assert node.parent_id is None
    assert node.bayesian_surprise is None


def test_node_variables_are_json():
    node = Node(
        id="abc", exploration_id="exp1", hypothesis="h",
        variables=json.dumps(["var1", "var2"]),
        relationships=json.dumps(["inversely proportional"]),
    )
    assert json.loads(node.variables) == ["var1", "var2"]


def test_belief_sample_creation():
    bs = BeliefSample(node_id="abc", phase="prior", sample_index=0, believes_hypothesis=True)
    assert bs.phase == "prior"
    assert bs.believes_hypothesis is True


def test_exploration_creation():
    exp = Exploration(id="exp1", domain="AI for science")
    assert exp.status == "initialized"


def test_agent_invocation_creation():
    inv = AgentInvocation(node_id="abc", role="experiment_generator", provider="claude")
    assert inv.exit_code is None
