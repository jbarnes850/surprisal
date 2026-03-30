import json
from surprisal.models import Node
from surprisal.export import export_json, export_csv, export_markdown, export_training_data


def _seed_nodes(db):
    """Insert test nodes with varying surprisal."""
    nodes = [
        Node(id="n1", exploration_id="e", hypothesis="High surprisal hypothesis",
             status="verified", bayesian_surprise=2.5,
             depth=1, context="test context", prior_alpha=23.0, prior_beta=9.0,
             posterior_alpha=9.0, posterior_beta=23.0, prior_mean=0.72, posterior_mean=0.28),
        Node(id="n2", exploration_id="e", hypothesis="Medium surprisal",
             status="verified", bayesian_surprise=1.2, depth=2),
        Node(id="n3", exploration_id="e", hypothesis="No surprisal",
             status="verified", bayesian_surprise=0.0, depth=1),
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
    assert "**Bayesian Surprise:** 2.500 (prior: 0.72 → posterior: 0.28)" in md


def test_export_training_data(tmp_db):
    _seed_nodes(tmp_db)
    jsonl = export_training_data(tmp_db)
    lines = jsonl.strip().split("\n")
    assert len(lines) == 3  # 3 verified nodes
    first = json.loads(lines[0])
    assert "hypothesis" in first
    assert "bayesian_surprise" in first
    assert first["bayesian_surprise"] >= 0


def test_export_json_includes_cited_papers(tmp_db):
    papers = json.dumps([{"arxiv_id": "2602.07670", "title": "Test Paper", "gap": "gap"}])
    tmp_db.insert_node(Node(
        id="n_lit", exploration_id="e", hypothesis="Literature-grounded hypothesis",
        status="verified", bayesian_surprise=5.0,
        depth=1, cited_papers=papers,
    ))
    result = export_json(tmp_db)
    hyp = result["hypotheses"][0]
    assert "cited_papers" in hyp


def test_export_markdown_zero_surprise_not_na(tmp_db):
    """BS=0.0 should render as '0.000', not 'N/A' (truthiness bug)."""
    _seed_nodes(tmp_db)
    md = export_markdown(tmp_db)
    assert "**Bayesian Surprise:** 0.000" in md


def test_export_markdown_includes_finding(tmp_db):
    tmp_db.insert_node(Node(
        id="n_finding", exploration_id="e",
        hypothesis="Token-loss statistics do not predict quality",
        finding="CV, skewness, kurtosis show |r| < 0.07. F-test p = 0.37.",
        status="verified", bayesian_surprise=4.1, depth=2,
        prior_mean=0.65, posterior_mean=0.45,
    ))
    md = export_markdown(tmp_db)
    assert "Token-loss statistics do not predict quality" in md
    assert "**Finding:** CV, skewness, kurtosis show |r| < 0.07" in md


def test_export_json_includes_finding(tmp_db):
    tmp_db.insert_node(Node(
        id="n_finding2", exploration_id="e",
        hypothesis="Short title",
        finding="Detailed result with p-values",
        status="verified", bayesian_surprise=1.0, depth=1,
    ))
    result = export_json(tmp_db)
    hyp = result["hypotheses"][0]
    assert hyp["finding"] == "Detailed result with p-values"


def test_export_markdown_includes_citations(tmp_db):
    papers = json.dumps([{"arxiv_id": "2602.07670", "title": "Test Paper", "gap": "untested"}])
    tmp_db.insert_node(Node(
        id="n_lit2", exploration_id="e", hypothesis="h",
        status="verified", bayesian_surprise=3.0,
        depth=1, cited_papers=papers,
    ))
    md = export_markdown(tmp_db)
    assert "2602.07670" in md
    assert "Test Paper" in md
