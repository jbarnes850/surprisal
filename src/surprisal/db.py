"""SQLite database layer for surprisal MCTS nodes and telemetry."""

import sqlite3
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Optional

from surprisal.models import AgentInvocation, BeliefSample, Node


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row

    def initialize(self):
        self.conn.execute("PRAGMA journal_mode=wal")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                exploration_id TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                parent_id TEXT,
                initial_hypothesis TEXT,
                context TEXT,
                variables TEXT,
                relationships TEXT,
                cited_papers TEXT,
                depth INTEGER DEFAULT 0,
                visit_count INTEGER DEFAULT 0,
                virtual_loss INTEGER DEFAULT 0,
                surprisal_sum REAL DEFAULT 0.0,
                bayesian_surprise REAL,
                belief_shifted BOOLEAN,
                prior_alpha REAL,
                prior_beta REAL,
                posterior_alpha REAL,
                posterior_beta REAL,
                k_prior INTEGER,
                k_post INTEGER,
                n_belief_samples INTEGER DEFAULT 30,
                status TEXT DEFAULT 'pending',
                branch_id TEXT,
                claude_session_id TEXT,
                codex_session_id TEXT,
                experiment_exit_code INTEGER,
                fsm_state TEXT DEFAULT 'start',
                fsm_failure_count INTEGER DEFAULT 0,
                fsm_revision_count INTEGER DEFAULT 0,
                dedup_cluster_id TEXT,
                created_at TIMESTAMP,
                verified_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS belief_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                sample_index INTEGER NOT NULL,
                believes_hypothesis BOOLEAN NOT NULL,
                raw_response TEXT,
                created_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS agent_invocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                role TEXT NOT NULL,
                provider TEXT NOT NULL,
                prompt_hash TEXT,
                response_hash TEXT,
                duration_seconds REAL,
                exit_code INTEGER,
                created_at TIMESTAMP
            );

            CREATE VIEW IF NOT EXISTS node_visit_stats AS
            SELECT
                c.id AS node_id,
                c.parent_id,
                c.visit_count,
                c.virtual_loss,
                c.surprisal_sum,
                CASE WHEN (c.visit_count + c.virtual_loss) > 0
                     THEN c.surprisal_sum / (c.visit_count + c.virtual_loss)
                     ELSE 0.0
                END AS exploit_score,
                COALESCE(p.visit_count, 0) AS parent_visits
            FROM nodes c
            LEFT JOIN nodes p ON c.parent_id = p.id;
            """
        )
        self.conn.commit()

        # Migration: add columns that may be missing in old databases
        existing_cols = {row[1] for row in self.execute("PRAGMA table_info(nodes)").fetchall()}
        if "cited_papers" not in existing_cols:
            self.execute("ALTER TABLE nodes ADD COLUMN cited_papers TEXT")
            self.conn.commit()

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    _NODE_COLUMNS = [f.name for f in dataclass_fields(Node)]

    def insert_node(self, node: Node):
        cols = self._NODE_COLUMNS
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        values = tuple(getattr(node, c) for c in cols)
        self.conn.execute(
            f"INSERT INTO nodes ({col_names}) VALUES ({placeholders})", values
        )
        self.conn.commit()

    def _row_to_node(self, row: sqlite3.Row) -> Node:
        d = dict(row)
        # SQLite stores booleans as 0/1 integers; convert back
        if d.get("belief_shifted") is not None:
            d["belief_shifted"] = bool(d["belief_shifted"])
        return Node(**d)

    def get_node(self, id: str) -> Optional[Node]:
        row = self.conn.execute("SELECT * FROM nodes WHERE id = ?", (id,)).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def update_node(self, id: str, **fields):
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [id]
        self.conn.execute(
            f"UPDATE nodes SET {set_clause} WHERE id = ?", values
        )
        self.conn.commit()

    def get_children(self, parent_id: str, exclude_pruned: bool = False) -> list[Node]:
        sql = "SELECT * FROM nodes WHERE parent_id = ?"
        params: list = [parent_id]
        if exclude_pruned:
            sql += " AND status != 'pruned'"
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_path_to_root(self, node_id: str) -> list[Node]:
        path = []
        current_id = node_id
        while current_id is not None:
            node = self.get_node(current_id)
            if node is None:
                break
            path.append(node)
            current_id = node.parent_id
        return path

    # ------------------------------------------------------------------
    # Belief samples
    # ------------------------------------------------------------------

    def insert_belief_sample(self, sample: BeliefSample):
        self.conn.execute(
            """INSERT INTO belief_samples
               (node_id, phase, sample_index, believes_hypothesis, raw_response, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                sample.node_id,
                sample.phase,
                sample.sample_index,
                sample.believes_hypothesis,
                sample.raw_response,
                sample.created_at,
            ),
        )
        self.conn.commit()

    def get_belief_samples(self, node_id: str, phase: str) -> list[BeliefSample]:
        rows = self.conn.execute(
            "SELECT * FROM belief_samples WHERE node_id = ? AND phase = ? ORDER BY sample_index",
            (node_id, phase),
        ).fetchall()
        return [
            BeliefSample(
                id=r["id"],
                node_id=r["node_id"],
                phase=r["phase"],
                sample_index=r["sample_index"],
                believes_hypothesis=bool(r["believes_hypothesis"]),
                raw_response=r["raw_response"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Agent invocations
    # ------------------------------------------------------------------

    def insert_agent_invocation(self, inv: AgentInvocation):
        self.conn.execute(
            """INSERT INTO agent_invocations
               (node_id, role, provider, prompt_hash, response_hash,
                duration_seconds, exit_code, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                inv.node_id,
                inv.role,
                inv.provider,
                inv.prompt_hash,
                inv.response_hash,
                inv.duration_seconds,
                inv.exit_code,
                inv.created_at,
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def count_completed(self, exploration_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE exploration_id = ? AND status IN ('verified', 'failed')",
            (exploration_id,),
        ).fetchone()
        return row[0]

    def count_completed_expansions(self, exploration_id: str) -> int:
        row = self.conn.execute(
            """SELECT COUNT(*) FROM nodes
               WHERE exploration_id = ? AND parent_id IS NOT NULL
               AND status IN ('verified', 'failed')""",
            (exploration_id,),
        ).fetchone()
        return row[0]

    def count_nodes(self, exploration_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE exploration_id = ?",
            (exploration_id,),
        ).fetchone()
        return row[0]

    def count_surprisals(self, exploration_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE exploration_id = ? AND belief_shifted = 1",
            (exploration_id,),
        ).fetchone()
        return row[0]

    def reset_stale_expanding(self, exploration_id: str | None = None):
        sql = "UPDATE nodes SET status = 'pending', virtual_loss = 0 WHERE status = 'expanding'"
        params = ()
        if exploration_id is not None:
            sql += " AND exploration_id = ?"
            params = (exploration_id,)
        self.conn.execute(sql, params)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Low-level
    # ------------------------------------------------------------------

    def execute(self, sql: str, params=None):
        if params is None:
            return self.conn.execute(sql)
        return self.conn.execute(sql, params)

    def close(self):
        self.conn.close()
