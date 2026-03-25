import numpy as np
import pytest
from autodiscovery.models import Node
from autodiscovery.dedup import (
    embed_hypothesis, build_embeddings, build_hac_tree,
    merge_decision, deduplicate,
)


def test_embed_hypothesis_deterministic():
    e1 = embed_hypothesis("test hypothesis")
    e2 = embed_hypothesis("test hypothesis")
    np.testing.assert_array_equal(e1, e2)


def test_embed_hypothesis_different_texts():
    e1 = embed_hypothesis("hypothesis A")
    e2 = embed_hypothesis("hypothesis B")
    assert not np.array_equal(e1, e2)


def test_embed_hypothesis_shape():
    e = embed_hypothesis("test")
    assert e.shape == (32,)
    assert e.dtype == np.float32


def test_build_hac_tree_returns_linkage():
    embeddings = np.random.rand(5, 32)
    Z = build_hac_tree(embeddings)
    assert Z is not None
    assert Z.shape[0] == 4  # n-1 merges for n=5


def test_build_hac_tree_too_few():
    embeddings = np.random.rand(1, 32)
    assert build_hac_tree(embeddings) is None


def test_merge_decision_above_threshold():
    assert merge_decision([True, True, True, True, True, True, True, True, False, False]) is True  # 8/10 = 0.8, above 0.7 threshold


def test_merge_decision_below_threshold():
    assert merge_decision([True, True, True, False, False, False, False, False, False, False]) is False  # 3/10


def test_merge_decision_empty():
    assert merge_decision([]) is False


def test_deduplicate_no_nodes(tmp_db):
    result = deduplicate(tmp_db)
    assert result["clusters"] == 0


def test_deduplicate_with_nodes(tmp_db):
    # Insert identical hypotheses — should cluster together
    for i in range(3):
        tmp_db.insert_node(Node(
            id=f"same_{i}", exploration_id="e",
            hypothesis="Identical hypothesis text",
            status="verified", bayesian_surprise=1.0,
        ))
    # Insert a different one
    tmp_db.insert_node(Node(
        id="diff", exploration_id="e",
        hypothesis="Completely different research direction about quantum computing",
        status="verified", bayesian_surprise=0.5,
    ))
    result = deduplicate(tmp_db)
    assert result["total_nodes"] == 4
    # The 3 identical should cluster; the different one should not
    assert result["clusters"] >= 1
