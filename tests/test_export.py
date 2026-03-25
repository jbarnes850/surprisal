import json
from autodiscovery.models import Node
from autodiscovery.export import export_json, export_csv, export_markdown, export_training_data


def _seed_nodes(db):
    """Insert test nodes with varying surprisal."""
    nodes = [
        Node(id="n1", exploration_id="e", hypothesis="High surprisal hypothesis",
             status="verified", bayesian_surprise=2.5, belief_shifted=True,
             depth=1, context="test context", prior_alpha=23.0, prior_beta=9.0,
             posterior_alpha=9.0, posterior_beta=23.0, k_prior=22, k_post=8),
        Node(id="n2", exploration_id="e", hypothesis="Medium surprisal",
             status="verified", bayesian_surprise=1.2, belief_shifted=True, depth=2),
        Node(id="n3", exploration_id="e", hypothesis="No surprisal",
             status="verified", bayesian_surprise=0.0, belief_shifted=False, depth=1),
        Node(id="n4", exploration_id="e", hypothesis="Failed node",
             status="failed", bayesian_surprise=None, depth=1),
    ]
    for n in nodes:
        db.insert_node(n)


def test_export_json_returns_all_verified(tmp_db):
    _seed_nodes(tmp_db)
    result = export_json(tmp_db)
    assert result["total"] == 3  # n1, n2, n3 (n4 is failed, excluded)
    assert result["hypotheses"][0]["bayesian_surprise"] == 2.5  # sorted desc


def test_export_json_top(tmp_db):
    _seed_nodes(tmp_db)
    result = export_json(tmp_db, top=1)
    assert result["total"] == 1
    assert result["hypotheses"][0]["id"] == "n1"


def test_export_json_min_surprisal(tmp_db):
    _seed_nodes(tmp_db)
    result = export_json(tmp_db, min_surprisal=1.0)
    assert result["total"] == 2  # n1 (2.5) and n2 (1.2)


def test_export_csv(tmp_db):
    _seed_nodes(tmp_db)
    csv_str = export_csv(tmp_db)
    assert "id,hypothesis" in csv_str
    assert "High surprisal" in csv_str
    lines = csv_str.strip().split("\n")
    assert len(lines) == 4  # header + 3 nodes


def test_export_markdown(tmp_db):
    _seed_nodes(tmp_db)
    md = export_markdown(tmp_db, top=2)
    assert "# AutoDiscovery Results" in md
    assert "High surprisal hypothesis" in md
    assert "**Bayesian Surprise:** 2.500" in md


def test_export_training_data(tmp_db):
    _seed_nodes(tmp_db)
    jsonl = export_training_data(tmp_db)
    lines = jsonl.strip().split("\n")
    assert len(lines) == 3  # 3 verified nodes
    first = json.loads(lines[0])
    assert "hypothesis" in first
    assert "surprisal" in first
    assert first["surprisal"] in (0, 1)
