"""Hypothesis deduplication via embeddings + hierarchical agglomerative clustering."""

import hashlib
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from surprisal.db import Database


def embed_hypothesis(text: str) -> np.ndarray:
    """Generate an embedding for a hypothesis text.
    Uses a deterministic hash-based embedding for fast deduplication.
    For higher-quality clustering, configure a real embedding endpoint."""
    h = hashlib.sha256(text.encode()).digest()
    # Convert 32 bytes to 32 floats in [-1, 1]
    arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
    return (arr / 128.0) - 1.0


def build_embeddings(db: Database, exploration_id: str | None = None) -> tuple[list[str], np.ndarray]:
    """Get embeddings for all verified hypotheses.
    Returns (node_ids, embeddings_matrix)."""
    sql = "SELECT id, hypothesis FROM nodes WHERE status = 'verified'"
    params: tuple[str, ...] = ()
    if exploration_id is not None:
        sql += " AND exploration_id = ?"
        params = (exploration_id,)
    rows = db.execute(sql, params).fetchall()
    if not rows:
        return [], np.array([])
    ids = [r[0] for r in rows]
    embeddings = np.array([embed_hypothesis(r[1]) for r in rows])
    return ids, embeddings


def build_hac_tree(embeddings: np.ndarray) -> Optional[np.ndarray]:
    """Build hierarchical agglomerative clustering tree.
    Returns scipy linkage matrix, or None if < 2 hypotheses."""
    if len(embeddings) < 2:
        return None
    distances = pdist(embeddings, metric="cosine")
    return linkage(distances, method="average")


def merge_decision(responses: list[bool], threshold: float = 0.7) -> bool:
    """Decide whether to merge based on LLM boolean responses.
    Merge if > threshold fraction say 'Yes'."""
    if not responses:
        return False
    yes_count = sum(1 for r in responses if r)
    return (yes_count / len(responses)) > threshold


def deduplicate(
    db: Database,
    max_distance: float = 0.5,
    exploration_id: str | None = None,
) -> dict:
    """Run deduplication on the hypothesis tree.
    Returns summary of clusters found."""
    node_ids, embeddings = build_embeddings(db, exploration_id=exploration_id)
    if len(node_ids) < 2:
        if exploration_id is None:
            db.execute("UPDATE nodes SET dedup_cluster_id = NULL")
        else:
            db.execute(
                "UPDATE nodes SET dedup_cluster_id = NULL WHERE exploration_id = ?",
                (exploration_id,),
            )
        db.conn.commit()
        return {"clusters": 0, "duplicates": 0, "total_nodes": len(node_ids)}

    Z = build_hac_tree(embeddings)
    if Z is None:
        return {"clusters": 0, "duplicates": 0, "total_nodes": len(node_ids)}

    # Cut the tree at the distance threshold
    labels = fcluster(Z, t=max_distance, criterion="distance")

    # Group by cluster
    clusters = {}
    for nid, label in zip(node_ids, labels):
        clusters.setdefault(int(label), []).append(nid)

    if exploration_id is None:
        db.execute("UPDATE nodes SET dedup_cluster_id = NULL")
    else:
        db.execute(
            "UPDATE nodes SET dedup_cluster_id = NULL WHERE exploration_id = ?",
            (exploration_id,),
        )

    # Mark duplicates in db
    duplicates = 0
    for cluster_id, members in clusters.items():
        if len(members) > 1:
            cluster_tag = f"cluster_{cluster_id}"
            for nid in members:
                db.update_node(nid, dedup_cluster_id=cluster_tag)
            duplicates += len(members) - 1  # all but one are duplicates
    db.conn.commit()

    return {
        "clusters": len([m for m in clusters.values() if len(m) > 1]),
        "duplicates": duplicates,
        "total_nodes": len(node_ids),
    }
