#!/usr/bin/env python3
"""
evaluate.py - Post-training evaluation for Red-vs-Blue agents
============================================================
Runs trained models for a fixed number of episodes and writes a CSV summary.
Default: 50 clean episodes (scenario=normal).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import config
from config import (
    MAX_STEPS_PER_EP,
    LOG_DIR,
    MODEL_DIR,
    RED_ACTIONS,
    BLUE_ACTIONS,
    SCENARIO_TYPES,
)
from environment import NetworkEnvironment
from red_agent import RedAgent
from blue_agent import BlueAgent


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for repeatability."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except Exception:
        pass


def normalize_scenario(value: str | None) -> str | None:
    """Normalize scenario input (clean -> normal, mixed -> random)."""
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"mixed", "random", "any", "all"}:
        return None
    if lowered == "clean":
        return "normal"
    return lowered


def resolve_output_path(path_arg: str | None, default_dir: str, filename: str) -> Path:
    """Resolve an output path, allowing either a directory or full file path."""
    if path_arg is None:
        return Path(default_dir) / filename
    candidate = Path(path_arg)
    if candidate.suffix:
        return candidate
    return candidate / filename


def _safe_num(value, fmt: str | None = None):
    """Return a safe CSV-friendly value (empty if None)."""
    if value is None:
        return ""
    if fmt is None:
        return value
    return format(value, fmt)


def load_agents(model_dir: str, epsilon: float) -> tuple[RedAgent, BlueAgent]:
    """Instantiate agents and load weights from disk."""
    red = RedAgent(epsilon=epsilon)
    blue = BlueAgent(epsilon=epsilon)

    red_path = Path(model_dir) / "red_model.pth"
    blue_path = Path(model_dir) / "blue_model.pth"
    if not red_path.is_file() or not blue_path.is_file():
        raise FileNotFoundError(
            f"Missing model files in {model_dir}: red_model.pth / blue_model.pth"
        )

    red.load(str(red_path))
    blue.load(str(blue_path))
    return red, blue


def save_action_stats(path: Path, atk_stats: dict, def_stats: dict) -> None:
    """Persist per-action success statistics as JSON."""
    data = {
        "attacks": {k: dict(v) for k, v in atk_stats.items()},
        "defenses": {k: dict(v) for k, v in def_stats.items()},
    }
    for category in data.values():
        for name, stats in category.items():
            attempts = stats["attempts"]
            stats["success_rate"] = round(
                stats["successes"] / attempts, 4
            ) if attempts > 0 else 0.0

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def evaluate(
    episodes: int,
    scenario: str | None,
    max_steps: int,
    epsilon: float,
    model_dir: str,
    output_csv: Path,
    stats_json: Path,
    seed: int | None,
) -> None:
    """Run evaluation episodes and write results to CSV."""
    if seed is not None:
        set_seed(seed)

    red_agent, blue_agent = load_agents(model_dir, epsilon)
    env = NetworkEnvironment()

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    csv_columns = [
        "episode",
        "scenario",
        "epsilon",
        "steps",
        "red_reward",
        "blue_reward",
        "red_wins",
        "blue_wins",
        "winner",
        "kill_chain_stage_reached",
        "chain_interrupted",
        "time_to_detect",
        "time_to_recover",
        "network_availability_pct",
        "false_positive_rate",
        "honeypot_triggered",
        "attack_types_used",
        "defense_types_used",
        "attack_success_counts",
        "defense_success_counts",
    ]

    atk_stats = defaultdict(lambda: {"attempts": 0, "successes": 0})
    def_stats = defaultdict(lambda: {"attempts": 0, "successes": 0})

    total_red_wins = 0
    total_blue_wins = 0
    all_red_rewards = []
    all_blue_rewards = []

    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()

        print(f"[+] Evaluation CSV -> {output_csv}")
        print(f"[+] Action stats   -> {stats_json}")

        for episode in range(1, episodes + 1):
            state, info = env.reset(scenario=scenario)
            scenario_name = info.get("scenario", "unknown")

            ep_red_reward = 0.0
            ep_blue_reward = 0.0

            for _ in range(max_steps):
                red_action = red_agent.act(state)
                next_state, red_reward, red_done = env.red_step(red_action)
                ep_red_reward += red_reward

                atk_name = RED_ACTIONS[red_action]["name"]
                atk_stats[atk_name]["attempts"] += 1
                if red_reward > 0:
                    atk_stats[atk_name]["successes"] += 1

                if red_done:
                    state = next_state
                    break

                blue_action = blue_agent.act(next_state)
                next_state2, blue_reward, blue_done = env.blue_step(blue_action)
                ep_blue_reward += blue_reward

                def_name = BLUE_ACTIONS[blue_action]["name"]
                def_stats[def_name]["attempts"] += 1
                if blue_reward > 0:
                    def_stats[def_name]["successes"] += 1

                state = next_state2

                if blue_done:
                    break

            metrics = env.episode_metrics
            total_steps = max(metrics["total_steps"], 1)

            avail_pct = (metrics["availability_steps"] / total_steps) * 100.0
            fp_rate = metrics["false_positives"] / total_steps

            kill_stage = max(metrics.get("kill_chain_stage_reached", 0), env.kill_chain_stage)

            red_won = ep_red_reward > ep_blue_reward
            blue_won = ep_blue_reward > ep_red_reward
            winner = "RED" if red_won else ("BLUE" if blue_won else "DRAW")

            total_red_wins += int(red_won)
            total_blue_wins += int(blue_won)
            all_red_rewards.append(ep_red_reward)
            all_blue_rewards.append(ep_blue_reward)

            row = {
                "episode": episode,
                "scenario": scenario_name,
                "epsilon": f"{red_agent.epsilon:.4f}",
                "steps": metrics["total_steps"],
                "red_reward": f"{ep_red_reward:.2f}",
                "blue_reward": f"{ep_blue_reward:.2f}",
                "red_wins": total_red_wins,
                "blue_wins": total_blue_wins,
                "winner": winner,
                "kill_chain_stage_reached": kill_stage,
                "chain_interrupted": int(metrics["chain_interrupted"]),
                "time_to_detect": _safe_num(metrics["time_to_detect"], "d"),
                "time_to_recover": _safe_num(metrics["time_to_recover"], "d"),
                "network_availability_pct": f"{avail_pct:.1f}",
                "false_positive_rate": f"{fp_rate:.4f}",
                "honeypot_triggered": int(metrics["honeypot_triggered"]),
                "attack_types_used": json.dumps(metrics["attack_types_used"]),
                "defense_types_used": json.dumps(metrics["defense_types_used"]),
                "attack_success_counts": json.dumps(metrics["attack_success_count"]),
                "defense_success_counts": json.dumps(metrics["defense_success_count"]),
            }
            writer.writerow(row)
            csv_file.flush()

            print(
                f"Ep {episode:3d}/{episodes} | {scenario_name:20s} | "
                f"Red: {ep_red_reward:+7.1f} | Blue: {ep_blue_reward:+7.1f} | "
                f"Winner: {winner}"
            )

    save_action_stats(stats_json, atk_stats, def_stats)

    avg_red = sum(all_red_rewards) / max(len(all_red_rewards), 1)
    avg_blue = sum(all_blue_rewards) / max(len(all_blue_rewards), 1)

    print("\n" + "=" * 65)
    print("  Evaluation Complete")
    print("=" * 65)
    print(f"  Episodes        : {episodes}")
    print(f"  Red wins        : {total_red_wins} ({100*total_red_wins/episodes:.1f}%)")
    print(f"  Blue wins       : {total_blue_wins} ({100*total_blue_wins/episodes:.1f}%)")
    print(f"  Avg Red reward  : {avg_red:+.2f}")
    print(f"  Avg Blue reward : {avg_blue:+.2f}")
    print(f"  CSV logs        : {output_csv}")
    print(f"  Action stats    : {stats_json}")
    print("=" * 65 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained Red/Blue agents and export a CSV report."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes (default: 50).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="normal",
        help=(
            "Scenario to evaluate: normal, already_compromised, high_traffic, "
            "partial_defense, stealth, or mixed (random)."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS_PER_EP,
        help=f"Max steps per episode (default: {MAX_STEPS_PER_EP}).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Exploration epsilon during evaluation (default: 0.0).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=MODEL_DIR,
        help=f"Directory containing model files (default: {MODEL_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path or directory (default: logs/).",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default=None,
        help="Output JSON path or directory for action stats.",
    )
    parser.add_argument(
        "--telemetry",
        choices=["influx", "synthetic"],
        default=None,
        help="Override TELEMETRY_MODE for this run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for repeatability.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.telemetry is not None:
        config.TELEMETRY_MODE = args.telemetry

    scenario = normalize_scenario(args.scenario)
    if scenario is not None and scenario not in SCENARIO_TYPES:
        raise ValueError(
            f"Invalid scenario '{args.scenario}'. Valid: {', '.join(SCENARIO_TYPES)}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = resolve_output_path(
        args.output, LOG_DIR, f"evaluation_{timestamp}.csv"
    )
    stats_json = resolve_output_path(
        args.stats_output, output_csv.parent, f"eval_action_stats_{timestamp}.json"
    )

    evaluate(
        episodes=args.episodes,
        scenario=scenario,
        max_steps=args.max_steps,
        epsilon=args.epsilon,
        model_dir=args.model_dir,
        output_csv=output_csv,
        stats_json=stats_json,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
