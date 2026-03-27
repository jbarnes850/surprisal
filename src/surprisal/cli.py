import json
import os
import sys
import uuid
import asyncio
from pathlib import Path

import click

from surprisal.config import AutoDiscoveryConfig, load_config, save_config
from surprisal.db import Database
from surprisal.models import Node
from surprisal.exploration import (
    create_exploration, load_exploration, find_latest_exploration,
)
from surprisal.orchestrator import run_exploration


def get_home() -> Path:
    if "SURPRISAL_HOME" in os.environ:
        return Path(os.environ["SURPRISAL_HOME"])
    return Path.home() / ".surprisal"


def get_config_path() -> Path:
    explicit = os.environ.get("SURPRISAL_CONFIG")
    if explicit:
        return Path(explicit)

    if "SURPRISAL_HOME" in os.environ:
        return get_home() / "config.toml"

    legacy_config = get_home() / "config.toml"
    xdg_root = Path(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")))
    xdg_config = xdg_root / "surprisal" / "config.toml"
    if legacy_config.exists() and not xdg_config.exists():
        return legacy_config
    return xdg_config


def get_config() -> AutoDiscoveryConfig:
    return load_config(get_config_path())


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return f"{'*' * max(len(value) - 4, 4)}{value[-4:]}"


def _resolve_exploration_dir(home: Path, exp_id: str | None) -> Path | None:
    if exp_id:
        exp_dir = home / exp_id
        return exp_dir if exp_dir.exists() else None
    return find_latest_exploration(home)


def _emit_error(message: str, as_json: bool = False, extra: dict | None = None) -> None:
    payload = {"status": "error", "message": message}
    if extra:
        payload.update(extra)
    if as_json:
        click.echo(json.dumps(payload))
    else:
        click.echo(f"Error: {message}", err=True)
    raise SystemExit(1)


def _run_explore_command(
    exp_id: str | None,
    budget: int | None,
    concurrency: int | None,
    c_explore: float | None,
    dry_run: bool,
    as_json: bool,
) -> None:
    home = get_home()
    cfg = get_config()

    exp_dir = _resolve_exploration_dir(home, exp_id)
    if not exp_dir or not exp_dir.exists():
        _emit_error("No exploration found. Run 'surprisal init --domain <topic>' first.", as_json=as_json)

    exp = load_exploration(exp_dir)
    budget = budget or cfg.general.default_budget
    concurrency = min(concurrency or cfg.general.default_concurrency, 4)
    c_explore = c_explore or cfg.mcts.c_explore

    db = Database(exp_dir / "tree.db")
    db.initialize()

    root = db.execute("SELECT id FROM nodes WHERE parent_id IS NULL LIMIT 1").fetchone()
    if not root:
        db.close()
        _emit_error("No root node. Run 'surprisal init' first.", as_json=as_json)

    if dry_run:
        from surprisal.mcts import select_node
        selected = select_node(db, root[0], c_explore,
                               cfg.mcts.k_progressive, cfg.mcts.alpha_progressive,
                               cfg.mcts.max_depth)
        node = db.get_node(selected)
        payload = {
            "status": "dry_run",
            "node_id": selected,
            "depth": node.depth,
            "hypothesis": node.hypothesis,
        }
        if as_json:
            click.echo(json.dumps(payload))
        else:
            click.echo(f"Would expand node {selected} at depth {node.depth}: {node.hypothesis}")
        db.close()
        return

    from surprisal.providers import detect_literature_provider, detect_providers, ensure_runner_auth

    providers = asyncio.run(detect_providers())
    if not providers.claude_available:
        db.close()
        extra = {"codex_available": providers.codex_available}
        _emit_error("Claude CLI is required. Run 'claude auth login' first.", as_json=as_json, extra=extra)

    # Ensure Docker runner can authenticate Claude inside the container
    if cfg.sandbox.backend in ("docker", "local"):
        from surprisal.config import save_config
        if not ensure_runner_auth(config_path=get_config_path(), save_config_fn=save_config, cfg=cfg):
            db.close()
            _emit_error(
                "Docker runner auth unavailable. Run 'claude setup-token' and try again.",
                as_json=as_json,
            )

    literature = asyncio.run(detect_literature_provider())
    progress_callback = None
    if not as_json:
        if providers.both_available:
            click.echo("Providers: Claude + Codex detected.")
        else:
            click.echo("Providers: Claude detected (Codex not found -- Claude will handle all roles).")
        click.echo(
            f"Literature: {literature.provider}"
            + (" (semantic search)" if literature.has_semantic_search else " (public API)")
        )

        def progress_callback(message: str) -> None:
            click.echo(f"Progress: {message}")

    result = asyncio.run(run_exploration(
        db=db, exploration_dir=exp_dir, budget=budget,
        concurrency=concurrency, c_explore=c_explore,
        config=cfg, root_id=root[0], domain=exp.domain,
        providers=providers, literature_provider=literature,
        progress_callback=progress_callback,
    ))
    db.close()

    if as_json:
        click.echo(json.dumps(result))
    else:
        click.echo(json.dumps(result, indent=2))

    if result.get("status") == "error":
        raise SystemExit(1)


@click.group()
def main():
    """surprisal -- open-ended scientific discovery via Bayesian surprise."""
    pass


@main.command()
@click.option("--domain", required=True, help="Research domain description")
@click.option("--seed", default=None, help="Seed hypothesis for root node")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def init(domain, seed, as_json):
    """Initialize a new exploration."""
    home = get_home()
    home.mkdir(parents=True, exist_ok=True)
    exp = create_exploration(home, domain=domain)

    # Create database and root node
    db_path = home / exp.id / "tree.db"
    db = Database(db_path)
    db.initialize()

    # Check if root already exists
    existing = db.execute("SELECT id FROM nodes WHERE parent_id IS NULL LIMIT 1").fetchone()
    if existing:
        root_id = existing[0]
    else:
        root_id = uuid.uuid4().hex[:12]
        root = Node(
            id=root_id,
            exploration_id=exp.id,
            hypothesis=seed or f"Initial research direction for: {domain}",
            depth=0,
            status="verified",
            visit_count=1,
            branch_id="root",
        )
        db.insert_node(root)

    db.close()

    if as_json:
        click.echo(json.dumps({"exploration_id": exp.id, "root_node_id": root_id}))
    else:
        click.echo(f"Exploration {exp.id} initialized for domain: {domain}")
        click.echo(f"Root node: {root_id}")


@main.command()
@click.option("--budget", default=None, type=int, help="Max MCTS iterations")
@click.option("--concurrency", default=None, type=int, help="Parallel workers (max 4)")
@click.option("--exploration", "exp_id", default=None, help="Exploration ID to run")
@click.option("--c-explore", default=None, type=float, help="UCT exploration constant")
@click.option("--dry-run", is_flag=True, help="Show what would be expanded")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def explore(budget, concurrency, exp_id, c_explore, dry_run, as_json):
    """Run MCTS exploration on the latest or a specific exploration."""
    _run_explore_command(exp_id, budget, concurrency, c_explore, dry_run, as_json)


@main.command()
@click.option("--tree", is_flag=True, help="Show hypothesis tree")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--exploration", "exp_id", default=None, help="Exploration ID")
def status(tree, as_json, exp_id):
    """Show exploration status."""
    home = get_home()
    if exp_id:
        exp_dir = home / exp_id
    else:
        exp_dir = find_latest_exploration(home)

    if not exp_dir or not exp_dir.exists():
        click.echo("Error: No exploration found. Run 'surprisal init --domain <topic>' first.", err=True)
        sys.exit(1)

    exp = load_exploration(exp_dir)
    db = Database(exp_dir / "tree.db")
    db.initialize()

    total = db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    verified = db.execute("SELECT COUNT(*) FROM nodes WHERE status='verified'").fetchone()[0]
    expanding = db.execute("SELECT COUNT(*) FROM nodes WHERE status='expanding'").fetchone()[0]
    failed = db.execute("SELECT COUNT(*) FROM nodes WHERE status='failed'").fetchone()[0]
    surprisals = db.execute("SELECT COUNT(*) FROM nodes WHERE belief_shifted=1").fetchone()[0]
    max_depth = db.execute("SELECT MAX(depth) FROM nodes").fetchone()[0] or 0

    if as_json:
        click.echo(json.dumps({
            "exploration_id": exp.id, "domain": exp.domain,
            "nodes_total": total, "verified": verified, "expanding": expanding,
            "failed": failed, "surprisals": surprisals, "max_depth": max_depth,
        }))
    else:
        click.echo(f"Exploration: {exp.id} ({exp.domain})")
        click.echo(f"Nodes: {total} total, {verified} verified, {expanding} expanding, {failed} failed")
        click.echo(f"Surprisals: {surprisals} found ({surprisals/max(verified,1)*100:.1f}% rate)")
        click.echo(f"Tree depth: max {max_depth}")

        papers_count = db.execute(
            "SELECT COUNT(*) FROM nodes WHERE cited_papers IS NOT NULL AND cited_papers != '[]' AND cited_papers != 'null'"
        ).fetchone()[0]
        if papers_count:
            click.echo(f"Literature-grounded: {papers_count} hypotheses with citations")

    if tree:
        _print_tree(db, None, 0)

    db.close()


def _print_tree(db, parent_id, indent):
    if parent_id is None:
        nodes = [n for n in db.execute("SELECT id, hypothesis, depth, bayesian_surprise, status FROM nodes WHERE parent_id IS NULL").fetchall()]
    else:
        nodes = [n for n in db.execute("SELECT id, hypothesis, depth, bayesian_surprise, status FROM nodes WHERE parent_id=?", (parent_id,)).fetchall()]
    for nid, hyp, depth, bs, st in nodes:
        bs_str = f" BS={bs:.2f}" if bs else ""
        marker = "!" if st == "failed" else ("*" if bs and bs > 0 else " ")
        click.echo(f"{'  ' * indent}{marker} [{depth}] {hyp[:80]}{bs_str}")
        _print_tree(db, nid, indent + 1)


@main.command()
@click.option("--format", "fmt", type=click.Choice(["json", "csv", "md"]), default="md")
@click.option("--top", default=None, type=int, help="Top N hypotheses")
@click.option("--min-surprisal", default=None, type=float)
@click.option("--training-data", is_flag=True, help="Export verified nodes as JSONL training data")
@click.option("--exploration", "exp_id", default=None)
@click.option("--json", "as_json", is_flag=True, help="Alias for --format json")
def export(fmt, top, min_surprisal, training_data, exp_id, as_json):
    """Export ranked hypotheses."""
    from surprisal.export import export_json, export_csv, export_markdown, export_training_data

    home = get_home()
    if exp_id:
        exp_dir = home / exp_id
    else:
        exp_dir = find_latest_exploration(home)

    if not exp_dir or not exp_dir.exists():
        click.echo("Error: No exploration found. Run 'surprisal init --domain <topic>' first.", err=True)
        sys.exit(1)

    db = Database(exp_dir / "tree.db")
    db.initialize()

    if training_data:
        click.echo(export_training_data(db))
    elif as_json:
        click.echo(json.dumps(export_json(db, top=top, min_surprisal=min_surprisal), indent=2))
    elif fmt == "json":
        click.echo(json.dumps(export_json(db, top=top, min_surprisal=min_surprisal), indent=2))
    elif fmt == "csv":
        click.echo(export_csv(db, top=top, min_surprisal=min_surprisal))
    elif fmt == "md":
        click.echo(export_markdown(db, top=top, min_surprisal=min_surprisal))

    db.close()


@main.command()
@click.argument("target", required=False)
@click.option("--budget", default=None, type=int, help="Max MCTS iterations")
@click.option("--concurrency", default=None, type=int, help="Parallel workers (max 4)")
@click.option("--c-explore", default=None, type=float, help="UCT exploration constant")
@click.option("--dry-run", is_flag=True, help="Show what would be expanded")
@click.option("--json", "as_json", is_flag=True)
def resume(target, budget, concurrency, c_explore, dry_run, as_json):
    """Resume the latest or a specific exploration."""
    _run_explore_command(target, budget, concurrency, c_explore, dry_run, as_json)


@main.command()
@click.option("--below-surprisal", default=0.05, type=float)
@click.option("--min-visits", default=3, type=int)
@click.option("--dry-run", is_flag=True)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def prune(below_surprisal, min_visits, dry_run, as_json):
    """Remove low-value branches."""
    home = get_home()
    exp_dir = find_latest_exploration(home)
    if not exp_dir:
        _emit_error("No exploration found.", as_json=as_json)

    db = Database(exp_dir / "tree.db")
    db.initialize()

    candidates = db.execute(
        """SELECT id, hypothesis, visit_count, surprisal_sum
           FROM nodes WHERE visit_count >= ? AND status != 'pruned'
           AND visit_count > 0 AND (surprisal_sum * 1.0 / visit_count) < ?""",
        (min_visits, below_surprisal),
    ).fetchall()

    if not candidates:
        if as_json:
            click.echo(json.dumps({"status": "noop", "pruned": [], "dry_run": dry_run}))
        else:
            click.echo("Nothing to prune.")
        db.close()
        return

    pruned = []
    for nid, hyp, vc, ss in candidates:
        avg = ss / vc if vc > 0 else 0
        item = {
            "node_id": nid,
            "hypothesis": hyp,
            "visit_count": vc,
            "avg_surprisal": avg,
        }
        pruned.append(item)
        if dry_run:
            if not as_json:
                click.echo(f"Would prune: {nid} -- {hyp[:60]} (avg surprisal: {avg:.3f})")
        else:
            db.update_node(nid, status="pruned")
            if not as_json:
                click.echo(f"Pruned: {nid} -- {hyp[:60]}")

    db.close()
    if as_json:
        click.echo(json.dumps({
            "status": "dry_run" if dry_run else "completed",
            "dry_run": dry_run,
            "pruned": pruned,
        }))


@main.command()
@click.option("--set", "set_val", nargs=2, help="Set key value")
@click.option("--show", is_flag=True, help="Show current config")
@click.option("--reset", is_flag=True, help="Reset to defaults")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def config(set_val, show, reset, as_json):
    """Manage configuration."""
    config_path = get_config_path()

    if reset:
        cfg = AutoDiscoveryConfig()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        save_config(cfg, config_path)
        if as_json:
            click.echo(json.dumps({"status": "reset", "config_path": str(config_path)}))
        else:
            click.echo("Config reset to defaults.")
        return

    cfg = load_config(config_path)

    if set_val:
        key, value = set_val
        try:
            cfg.set(key, value)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            save_config(cfg, config_path)
            if as_json:
                click.echo(json.dumps({
                    "status": "updated",
                    "config_path": str(config_path),
                    "key": key,
                    "value": _mask_secret(value) if key.startswith("credentials.") else value,
                }))
            else:
                click.echo(f"Set {key} = {value}")
        except KeyError as e:
            _emit_error(str(e), as_json=as_json)
        return

    if show:
        from dataclasses import fields as dc_fields
        sections = {}
        for section_name in ("general", "mcts", "agents", "sandbox", "credentials"):
            section = getattr(cfg, section_name)
            sections[section_name] = {}
            for f in dc_fields(section):
                value = getattr(section, f.name)
                if section_name == "credentials":
                    value = _mask_secret(value)
                sections[section_name][f.name] = value
        if as_json:
            click.echo(json.dumps({"config_path": str(config_path), "config": sections}, indent=2))
        else:
            click.echo(f"Config path: {config_path}")
            for section_name, values in sections.items():
                click.echo(f"[{section_name}]")
                for field_name, value in values.items():
                    click.echo(f"  {field_name} = {value}")
        return

    if as_json:
        click.echo(json.dumps({"status": "noop", "message": "Use --show to display config or --set key value to change a setting."}))
    else:
        click.echo("Use --show to display config or --set key value to change a setting.")
