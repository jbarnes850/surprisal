import pytest
import math
from autodiscovery.models import Node
from autodiscovery.mcts import compute_uct, select_node, backpropagate, max_children


def test_uct_unvisited_is_infinity():
    score = compute_uct(visit_count=0, virtual_loss=0, surprisal_sum=0.0, parent_visits=10, c_explore=1.414)
    assert score == float("inf")


def test_uct_exploit_only():
    score = compute_uct(visit_count=10, virtual_loss=0, surprisal_sum=5.0, parent_visits=100, c_explore=0.0)
    assert score == pytest.approx(0.5)


def test_uct_with_virtual_loss():
    score_without = compute_uct(visit_count=5, virtual_loss=0, surprisal_sum=2.0, parent_visits=20, c_explore=1.414)
    score_with = compute_uct(visit_count=5, virtual_loss=2, surprisal_sum=2.0, parent_visits=20, c_explore=1.414)
    assert score_with < score_without


def test_uct_explore_term_uses_factor_of_2():
    score = compute_uct(visit_count=1, virtual_loss=0, surprisal_sum=0.0, parent_visits=int(math.e**2), c_explore=1.0)
    expected_explore = math.sqrt(2 * math.log(int(math.e**2)) / 1)
    assert score == pytest.approx(expected_explore, abs=0.1)


def test_max_children_progressive_widening():
    assert max_children(visits=1, k=1.0, alpha=0.5) == 1
    assert max_children(visits=4, k=1.0, alpha=0.5) == 2
    assert max_children(visits=9, k=1.0, alpha=0.5) == 3
    assert max_children(visits=16, k=1.0, alpha=0.5) == 4
    assert max_children(visits=0, k=1.0, alpha=0.5) == 0


def test_select_node_returns_root_when_no_children(tmp_db):
    tmp_db.insert_node(Node(id="root", exploration_id="e", hypothesis="root", visit_count=1))
    selected = select_node(tmp_db, "root", c_explore=1.414, k=1.0, alpha=0.5)
    assert selected == "root"


def test_select_node_returns_root_when_widening_allows(tmp_db):
    tmp_db.insert_node(Node(id="root", exploration_id="e", hypothesis="root", visit_count=4))
    tmp_db.insert_node(Node(id="c1", exploration_id="e", hypothesis="c1", parent_id="root", depth=1, visit_count=1))
    selected = select_node(tmp_db, "root", c_explore=1.414, k=1.0, alpha=0.5)
    assert selected == "root"


def test_select_node_descends_when_widening_full(tmp_db):
    tmp_db.insert_node(Node(id="root", exploration_id="e", hypothesis="root", visit_count=1))
    tmp_db.insert_node(Node(id="c1", exploration_id="e", hypothesis="c1", parent_id="root", depth=1, visit_count=0))
    selected = select_node(tmp_db, "root", c_explore=1.414, k=1.0, alpha=0.5)
    assert selected == "c1"


def test_select_node_respects_max_depth(tmp_db):
    tmp_db.insert_node(Node(id="r", exploration_id="e", hypothesis="r", visit_count=100))
    parent_id = "r"
    for i in range(30):
        nid = f"d{i}"
        tmp_db.insert_node(Node(id=nid, exploration_id="e", hypothesis=f"h{i}", parent_id=parent_id, depth=i+1, visit_count=100))
        parent_id = nid
    selected = select_node(tmp_db, "r", c_explore=1.414, k=1.0, alpha=0.5, max_depth=30)
    node = tmp_db.get_node(selected)
    assert node.depth <= 30


def test_backpropagate_updates_ancestors(tmp_db):
    tmp_db.insert_node(Node(id="r", exploration_id="e", hypothesis="r"))
    tmp_db.insert_node(Node(id="a", exploration_id="e", hypothesis="a", parent_id="r", depth=1))
    tmp_db.insert_node(Node(id="b", exploration_id="e", hypothesis="b", parent_id="a", depth=2))
    backpropagate(tmp_db, "b", surprisal_value=1)
    assert tmp_db.get_node("b").visit_count == 1
    assert tmp_db.get_node("b").surprisal_sum == 1.0
    assert tmp_db.get_node("a").visit_count == 1
    assert tmp_db.get_node("a").surprisal_sum == 1.0
    assert tmp_db.get_node("r").visit_count == 1
    assert tmp_db.get_node("r").surprisal_sum == 1.0


def test_backpropagate_zero_surprisal(tmp_db):
    tmp_db.insert_node(Node(id="r", exploration_id="e", hypothesis="r"))
    tmp_db.insert_node(Node(id="a", exploration_id="e", hypothesis="a", parent_id="r", depth=1))
    backpropagate(tmp_db, "a", surprisal_value=0)
    assert tmp_db.get_node("a").visit_count == 1
    assert tmp_db.get_node("a").surprisal_sum == 0.0
    assert tmp_db.get_node("r").visit_count == 1
