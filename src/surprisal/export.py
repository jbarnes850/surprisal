import csv
import io
import json
from surprisal.db import Database


def export_json(db: Database, top: int = None, min_surprisal: float = None) -> dict:
    """Export hypotheses as JSON, ranked by bayesian_surprise descending."""
    nodes = _query_verified(db, top, min_surprisal)
    return {
        "hypotheses": [
            {
                "id": n.id,
                "hypothesis": n.hypothesis,
                "finding": n.finding,
                "initial_hypothesis": n.initial_hypothesis,
                "context": n.context,
                "variables": n.variables,
                "relationships": n.relationships,
                "depth": n.depth,
                "bayesian_surprise": n.bayesian_surprise,
                "prior_alpha": n.prior_alpha,
                "prior_beta": n.prior_beta,
                "posterior_alpha": n.posterior_alpha,
                "posterior_beta": n.posterior_beta,
                "prior_mean": n.prior_mean,
                "posterior_mean": n.posterior_mean,
                "cited_papers": n.cited_papers,
                "status": n.status,
            }
            for n in nodes
        ],
        "total": len(nodes),
    }


def export_csv(db: Database, top: int = None, min_surprisal: float = None) -> str:
    """Export hypotheses as CSV string."""
    nodes = _query_verified(db, top, min_surprisal)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "hypothesis", "depth", "bayesian_surprise", "prior_mean", "posterior_mean", "context"])
    for n in nodes:
        writer.writerow([n.id, n.hypothesis, n.depth, n.bayesian_surprise, n.prior_mean, n.posterior_mean, n.context])
    return output.getvalue()


def export_markdown(db: Database, top: int = None, min_surprisal: float = None) -> str:
    """Export hypotheses as a markdown report."""
    nodes = _query_verified(db, top, min_surprisal)
    lines = ["# AutoDiscovery Results", ""]
    lines.append(f"**Total hypotheses:** {len(nodes)}")
    lines.append("")
    for i, n in enumerate(nodes, 1):
        bs = f"{n.bayesian_surprise:.3f}" if n.bayesian_surprise is not None else "N/A"
        prior_str = f"{n.prior_mean:.2f}" if n.prior_mean is not None else "N/A"
        posterior_str = f"{n.posterior_mean:.2f}" if n.posterior_mean is not None else "N/A"
        lines.append(f"## {i}. {n.hypothesis}")
        lines.append("")
        if n.finding:
            lines.append(f"**Finding:** {n.finding}")
        lines.append(f"**Bayesian Surprise:** {bs} (prior: {prior_str} → posterior: {posterior_str})")
        lines.append(f"**Depth:** {n.depth}")
        if n.cited_papers:
            try:
                papers = json.loads(n.cited_papers)
                if papers:
                    lines.append("**Grounded in:**")
                    for p in papers:
                        lines.append(f"- [{p.get('arxiv_id', '?')}] \"{p.get('title', '?')}\" -- Gap: {p.get('gap', '?')}")
            except (json.JSONDecodeError, TypeError):
                pass
        if n.context:
            lines.append(f"**Context:** {n.context}")
        lines.append("")
    return "\n".join(lines)


def export_training_data(db: Database) -> str:
    """Export all verified nodes as JSONL for surprisal predictor training."""
    nodes = _query_verified(db, top=None, min_surprisal=None)
    lines = []
    for n in nodes:
        sample = {
            "hypothesis": n.hypothesis,
            "context": n.context,
            "variables": n.variables,
            "relationships": n.relationships,
            "depth": n.depth,
            "bayesian_surprise": n.bayesian_surprise,
            "prior_alpha": n.prior_alpha,
            "prior_beta": n.prior_beta,
            "posterior_alpha": n.posterior_alpha,
            "posterior_beta": n.posterior_beta,
        }
        lines.append(json.dumps(sample))
    return "\n".join(lines)


def _query_verified(db: Database, top: int = None, min_surprisal: float = None):
    """Query verified nodes, sorted by bayesian_surprise descending."""
    query = "SELECT * FROM nodes WHERE status = 'verified' AND bayesian_surprise IS NOT NULL"
    params = []
    if min_surprisal is not None:
        query += " AND bayesian_surprise > ?"
        params.append(min_surprisal)
    query += " ORDER BY bayesian_surprise DESC"
    if top:
        query += " LIMIT ?"
        params.append(top)
    rows = db.execute(query, tuple(params)).fetchall()
    # Convert rows to Node objects via db's row converter
    results = []
    for row in rows:
        node = db.get_node(row[0])  # row[0] is the id
        results.append(node)
    return results
