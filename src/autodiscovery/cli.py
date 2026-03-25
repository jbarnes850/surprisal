import json
import os
import sys
import uuid
import asyncio
import click
from pathlib import Path
from autodiscovery.config import AutoDiscoveryConfig, load_config, save_config
from autodiscovery.db import Database
from autodiscovery.models import Node
from autodiscovery.exploration import (
    create_exploration, load_exploration, find_latest_exploration,
)
from autodiscovery.orchestrator import run_exploration


def get_home() -> Path:
    return Path(os.environ.get("AUTODISCOVERY_HOME", os.path.expanduser("~/.autodiscovery")))


def get_config() -> AutoDiscoveryConfig:
    return load_config(get_home() / "config.toml")


@click.group()
def main():
    """autodiscovery -- open-ended scientific discovery via Bayesian surprise."""
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
@click.option("--exploration", "exp_id", default=None, help="Exploration ID to resume")
@click.option("--c-explore", default=None, type=float, help="UCT exploration constant")
@click.option("--dry-run", is_flag=True, help="Show what would be expanded")
def explore(budget, concurrency, exp_id, c_explore, dry_run):
    """Start or resume MCTS exploration."""
    home = get_home()
    cfg = get_config()

    if exp_id:
        exp_dir = home / exp_id
    else:
        exp_dir = find_latest_exploration(home)

    if not exp_dir or not exp_dir.exists():
        click.echo("Error: No exploration found. Run 'autodiscovery init --domain <topic>' first.", err=True)
        sys.exit(1)

    exp = load_exploration(exp_dir)
    budget = budget or cfg.general.default_budget
    concurrency = min(concurrency or cfg.general.default_concurrency, 4)
    c_explore = c_explore or cfg.mcts.c_explore

    db = Database(exp_dir / "tree.db")
    db.initialize()

    root = db.execute("SELECT id FROM nodes WHERE parent_id IS NULL LIMIT 1").fetchone()
    if not root:
        click.echo("Error: No root node. Run 'autodiscovery init' first.", err=True)
        sys.exit(1)

    if dry_run:
        from autodiscovery.mcts import select_node
        selected = select_node(db, root[0], c_explore,
                               cfg.mcts.k_progressive, cfg.mcts.alpha_progressive,
                               cfg.mcts.max_depth)
        node = db.get_node(selected)
        click.echo(f"Would expand node {selected} at depth {node.depth}: {node.hypothesis}")
        db.close()
        return

    result = asyncio.run(run_exploration(
        db=db, exploration_dir=exp_dir, budget=budget,
        concurrency=concurrency, c_explore=c_explore,
        config=cfg, root_id=root[0], domain=exp.domain,
    ))
    db.close()
    click.echo(json.dumps(result, indent=2))


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
        click.echo("Error: No exploration found. Run 'autodiscovery init --domain <topic>' first.", err=True)
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
@click.option("--training-data", is_flag=True, help="Export as JSONL for surprisal predictor")
@click.option("--exploration", "exp_id", default=None)
def export(fmt, top, min_surprisal, training_data, exp_id):
    """Export ranked hypotheses."""
    from autodiscovery.export import export_json, export_csv, export_markdown, export_training_data

    home = get_home()
    if exp_id:
        exp_dir = home / exp_id
    else:
        exp_dir = find_latest_exploration(home)

    if not exp_dir or not exp_dir.exists():
        click.echo("Error: No exploration found. Run 'autodiscovery init --domain <topic>' first.", err=True)
        sys.exit(1)

    db = Database(exp_dir / "tree.db")
    db.initialize()

    if training_data:
        click.echo(export_training_data(db))
    elif fmt == "json":
        click.echo(json.dumps(export_json(db, top=top, min_surprisal=min_surprisal), indent=2))
    elif fmt == "csv":
        click.echo(export_csv(db, top=top, min_surprisal=min_surprisal))
    elif fmt == "md":
        click.echo(export_markdown(db, top=top, min_surprisal=min_surprisal))

    db.close()


@main.command()
@click.argument("target", required=False)
@click.option("--json", "as_json", is_flag=True)
def resume(target, as_json):
    """Resume a specific branch or exploration."""
    home = get_home()
    if target:
        exp_dir = home / target
        if not exp_dir.exists():
            click.echo(f"Error: Branch or exploration '{target}' not found. Run 'autodiscovery status --tree' to see available branches.", err=True)
            sys.exit(1)
    else:
        exp_dir = find_latest_exploration(home)
        if not exp_dir:
            click.echo("Error: No explorations found. Run 'autodiscovery init --domain <topic>' first.", err=True)
            sys.exit(1)

    click.echo(f"Resuming exploration at {exp_dir}")


@main.command()
@click.option("--below-surprisal", default=0.05, type=float)
@click.option("--min-visits", default=3, type=int)
@click.option("--dry-run", is_flag=True)
def prune(below_surprisal, min_visits, dry_run):
    """Remove low-value branches."""
    home = get_home()
    exp_dir = find_latest_exploration(home)
    if not exp_dir:
        click.echo("Error: No exploration found.", err=True)
        sys.exit(1)

    db = Database(exp_dir / "tree.db")
    db.initialize()

    candidates = db.execute(
        """SELECT id, hypothesis, visit_count, surprisal_sum
           FROM nodes WHERE visit_count >= ? AND status != 'pruned'
           AND visit_count > 0 AND (surprisal_sum * 1.0 / visit_count) < ?""",
        (min_visits, below_surprisal),
    ).fetchall()

    if not candidates:
        click.echo("Nothing to prune.")
        db.close()
        return

    for nid, hyp, vc, ss in candidates:
        avg = ss / vc if vc > 0 else 0
        if dry_run:
            click.echo(f"Would prune: {nid} -- {hyp[:60]} (avg surprisal: {avg:.3f})")
        else:
            db.update_node(nid, status="pruned")
            click.echo(f"Pruned: {nid} -- {hyp[:60]}")

    db.close()


@main.command()
@click.option("--set", "set_val", nargs=2, help="Set key value")
@click.option("--show", is_flag=True, help="Show current config")
@click.option("--reset", is_flag=True, help="Reset to defaults")
def config(set_val, show, reset):
    """Manage configuration."""
    home = get_home()
    config_path = home / "config.toml"

    if reset:
        cfg = AutoDiscoveryConfig()
        home.mkdir(parents=True, exist_ok=True)
        save_config(cfg, config_path)
        click.echo("Config reset to defaults.")
        return

    cfg = load_config(config_path)

    if set_val:
        key, value = set_val
        try:
            cfg.set(key, value)
            home.mkdir(parents=True, exist_ok=True)
            save_config(cfg, config_path)
            click.echo(f"Set {key} = {value}")
        except KeyError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        return

    if show:
        from dataclasses import fields as dc_fields
        for section_name in ("general", "mcts", "agents", "sandbox", "predictor"):
            section = getattr(cfg, section_name)
            click.echo(f"[{section_name}]")
            for f in dc_fields(section):
                click.echo(f"  {f.name} = {getattr(section, f.name)}")
        return

    click.echo("Use --show to display config or --set key value to change a setting.")
