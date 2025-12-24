#!/usr/bin/env python3
"""
Experiment Runner for LLM Poker Arena

Runs a fair, balanced head-to-head round robin where every model
plays every other model. Adjust HANDS_PER_MATCHUP below to control
experiment size.

Usage:
    python run_experiments.py              # Run full experiment
    python run_experiments.py --resume     # Resume interrupted experiment
    python run_experiments.py --dry-run    # Show plan without executing
    python run_experiments.py --status     # Show current progress
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path
from threading import Lock

# One flagship model per lab (6 models, 15 matchups)
ALL_MODELS = [
    "sonnet",       # Anthropic
    "gpt5",         # OpenAI
    "deepseek",     # DeepSeek
    "mistral",      # Mistral
    "grok",         # xAI
    "gemini",       # Google
]

# Experiment configuration
HANDS_PER_MATCHUP = 1000

# State file for resumption
STATE_FILE = Path("data/experiment_state.json")

# Lock for thread-safe state updates
state_lock = Lock()


def get_all_matchups() -> list[tuple[str, str]]:
    """Generate all unique head-to-head matchups."""
    return list(combinations(ALL_MODELS, 2))


def load_state() -> dict:
    """Load experiment state from file."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "started_at": None,
        "completed_matchups": [],
        "in_progress": None,
        "total_hands": 0,
        "total_cost": 0.0,
        "errors": [],
    }


def save_state(state: dict) -> None:
    """Save experiment state to file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def run_matchup(m1: str, m2: str, hands: int, wandb: bool, quiet: bool) -> tuple[bool, float]:
    """Run a single head-to-head matchup."""
    cmd = [
        sys.executable, "main.py",
        "-n", str(hands),
        "-p", "2",
        "--models", m1, m2,
    ]
    if wandb:
        cmd.append("--wandb")
    if quiet:
        cmd.append("-q")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse cost from output
        cost = 0.0
        for line in result.stdout.split("\n"):
            if "Total estimated cost:" in line:
                try:
                    cost = float(line.split("$")[1].strip())
                except (IndexError, ValueError):
                    pass
                break

        if result.returncode != 0:
            print(f"\n    stderr: {result.stderr[:200]}")

        return result.returncode == 0, cost

    except Exception as e:
        print(f"\n    Exception: {e}")
        return False, 0.0


def show_status(state: dict) -> None:
    """Display current experiment status."""
    all_matchups = get_all_matchups()
    completed = len(state["completed_matchups"])
    total = len(all_matchups)
    remaining = total - completed

    print("\n" + "=" * 50)
    print("EXPERIMENT STATUS")
    print("=" * 50)

    if state["started_at"]:
        print(f"Started: {state['started_at']}")
    else:
        print("Not started yet")
        return

    print(f"\nProgress: {completed}/{total} matchups ({completed/total*100:.1f}%)")
    print(f"Remaining: {remaining} matchups")
    print(f"Hands played: {state['total_hands']:,}")
    print(f"Total cost: ${state['total_cost']:.2f}")

    if state["errors"]:
        print(f"Errors: {len(state['errors'])}")

    if state["in_progress"]:
        print(f"\nIn progress: {state['in_progress'][0]} vs {state['in_progress'][1]}")

    # Model stats
    print("\n" + "-" * 50)
    print("HANDS PER MODEL:")
    model_hands = {m: 0 for m in ALL_MODELS}
    for m1, m2 in state["completed_matchups"]:
        model_hands[m1] += HANDS_PER_MATCHUP
        model_hands[m2] += HANDS_PER_MATCHUP

    hands_per_model = HANDS_PER_MATCHUP * (len(ALL_MODELS) - 1)
    for model in ALL_MODELS:
        bar_len = int(model_hands[model] / hands_per_model * 20) if hands_per_model > 0 else 0
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"  {model:15} [{bar}] {model_hands[model]:,}/{hands_per_model:,}")


def show_plan() -> None:
    """Display the experiment plan."""
    matchups = get_all_matchups()

    print("\n" + "=" * 50)
    print("EXPERIMENT PLAN")
    print("=" * 50)
    num_opponents = len(ALL_MODELS) - 1
    hands_per_model = HANDS_PER_MATCHUP * num_opponents
    print(f"\nModels: {len(ALL_MODELS)}")
    print(f"Matchups: {len(matchups)}")
    print(f"Hands per matchup: {HANDS_PER_MATCHUP:,}")
    print(f"Total hands: {len(matchups) * HANDS_PER_MATCHUP:,}")
    print(f"Hands per model: {hands_per_model:,} (plays {num_opponents} opponents x {HANDS_PER_MATCHUP:,})")

    print("\n" + "-" * 50)
    print("ALL MATCHUPS:")
    for i, (m1, m2) in enumerate(matchups, 1):
        print(f"  {i:2}. {m1:15} vs {m2}")


def run_single_matchup(args: tuple) -> tuple:
    """Worker function for parallel execution."""
    m1, m2, hands, wandb, quiet, idx, total = args
    start_time = time.time()
    success, cost = run_matchup(m1, m2, hands, wandb, quiet)
    elapsed = time.time() - start_time
    return (m1, m2, success, cost, elapsed, idx, total)


def run_experiment(wandb: bool = True, quiet: bool = True, resume: bool = False, parallel: int = 1) -> None:
    """Run the full experiment."""
    state = load_state()

    if not resume and state["started_at"] and state["completed_matchups"]:
        all_matchups = get_all_matchups()
        print("Previous experiment found!")
        print(f"  Completed: {len(state['completed_matchups'])}/{len(all_matchups)} matchups")
        print(f"  Use --resume to continue, or delete data/experiment_state.json to restart")
        return

    if not state["started_at"]:
        state["started_at"] = datetime.now().isoformat()
        save_state(state)

    all_matchups = get_all_matchups()
    completed_set = {tuple(m) for m in state["completed_matchups"]}
    remaining = [(m1, m2) for m1, m2 in all_matchups if (m1, m2) not in completed_set]

    print("\n" + "=" * 60)
    print("LLM POKER ARENA - HEAD-TO-HEAD ROUND ROBIN")
    print("=" * 60)
    print(f"Total matchups: {len(all_matchups)}")
    print(f"Hands per matchup: {HANDS_PER_MATCHUP:,}")
    print(f"Total hands: {len(all_matchups) * HANDS_PER_MATCHUP:,}")
    print(f"Completed: {len(completed_set)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Parallel workers: {parallel}")
    print("=" * 60)

    if parallel == 1:
        # Sequential execution (original behavior)
        for i, (m1, m2) in enumerate(all_matchups, 1):
            if (m1, m2) in completed_set:
                continue

            print(f"\n[{i:2}/{len(all_matchups)}] {m1} vs {m2}")
            print(f"  Running {HANDS_PER_MATCHUP:,} hands...", end=" ", flush=True)

            start_time = time.time()
            success, cost = run_matchup(m1, m2, HANDS_PER_MATCHUP, wandb, quiet)
            elapsed = time.time() - start_time

            if success:
                print(f"Done! ({elapsed/60:.1f}min, ${cost:.2f})")
                state["completed_matchups"].append([m1, m2])
                state["total_hands"] += HANDS_PER_MATCHUP
                state["total_cost"] += cost
            else:
                print(f"FAILED ({elapsed/60:.1f}min)")
                state["errors"].append({
                    "matchup": [m1, m2],
                    "time": datetime.now().isoformat(),
                })

            save_state(state)

            if len(state["completed_matchups"]) % 5 == 0:
                pct = len(state["completed_matchups"]) / len(all_matchups) * 100
                print(f"\n  === Progress: {len(state['completed_matchups'])}/{len(all_matchups)} ({pct:.0f}%) | Cost: ${state['total_cost']:.2f} ===")
    else:
        # Parallel execution
        print(f"\nStarting {len(remaining)} matchups with {parallel} parallel workers...\n")

        # Prepare work items
        work_items = [
            (m1, m2, HANDS_PER_MATCHUP, wandb, quiet, i, len(all_matchups))
            for i, (m1, m2) in enumerate(remaining, len(completed_set) + 1)
        ]

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(run_single_matchup, item): item for item in work_items}

            for future in as_completed(futures):
                m1, m2, success, cost, elapsed, idx, total = future.result()

                with state_lock:
                    if success:
                        print(f"[{idx:2}/{total}] {m1} vs {m2}: Done! ({elapsed/60:.1f}min, ${cost:.2f})")
                        state["completed_matchups"].append([m1, m2])
                        state["total_hands"] += HANDS_PER_MATCHUP
                        state["total_cost"] += cost
                    else:
                        print(f"[{idx:2}/{total}] {m1} vs {m2}: FAILED ({elapsed/60:.1f}min)")
                        state["errors"].append({
                            "matchup": [m1, m2],
                            "time": datetime.now().isoformat(),
                        })
                    save_state(state)

                # Progress update
                completed = len(state["completed_matchups"])
                if completed % 5 == 0:
                    pct = completed / len(all_matchups) * 100
                    print(f"\n  === Progress: {completed}/{len(all_matchups)} ({pct:.0f}%) | Cost: ${state['total_cost']:.2f} ===\n")

    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Total matchups: {len(state['completed_matchups'])}")
    print(f"Total hands: {state['total_hands']:,}")
    print(f"Total cost: ${state['total_cost']:.2f}")
    if state["errors"]:
        print(f"Failed matchups: {len(state['errors'])}")
        for err in state["errors"]:
            print(f"  - {err['matchup'][0]} vs {err['matchup'][1]}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM Poker experiments")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted experiment")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--status", action="store_true", help="Show current progress")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--verbose", action="store_true", help="Show game output")
    parser.add_argument("--parallel", "-p", type=int, default=1,
                        help="Number of matchups to run in parallel (default: 1)")
    args = parser.parse_args()

    if args.status:
        show_status(load_state())
    elif args.dry_run:
        show_plan()
    else:
        try:
            run_experiment(
                wandb=not args.no_wandb,
                quiet=not args.verbose,
                resume=args.resume,
                parallel=args.parallel,
            )
        except KeyboardInterrupt:
            print("\n\nInterrupted! Progress saved. Use --resume to continue.")


if __name__ == "__main__":
    main()
