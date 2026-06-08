#!/usr/bin/env python3
"""
demo.py - Live demo runner for trained Red-vs-Blue agents
========================================================
Runs one or more episodes with detailed step-by-step console output and
optional JSON logging for presentation recording.
"""

from __future__ import annotations

import argparse
import json
import random
import time
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a live demo using trained Red/Blue models."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of demo episodes (default: 1).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="normal",
        help=(
            "Scenario to run: normal, already_compromised, high_traffic, "
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
        help="Exploration epsilon during demo (default: 0.0).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=MODEL_DIR,
        help=f"Directory containing model files (default: {MODEL_DIR}).",
    )
    parser.add_argument(
        "--telemetry",
        choices=["influx", "synthetic"],
        default=None,
        help="Override TELEMETRY_MODE for this run.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=None,
        help="Override environment step delay (seconds).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Extra sleep after each Blue step (seconds).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for repeatability.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Optional JSON log path (default: logs/demo_<timestamp>.json).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable JSON logging.",
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

    if args.seed is not None:
        set_seed(args.seed)

    red_agent, blue_agent = load_agents(args.model_dir, args.epsilon)

    env = NetworkEnvironment(step_delay=args.step_delay)

    log_path = None
    if not args.no_log:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.log:
            log_path = Path(args.log)
        else:
            log_path = Path(LOG_DIR) / f"demo_{timestamp}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)

    demo_log = []

    for ep in range(1, args.episodes + 1):
        state, info = env.reset(scenario=scenario)
        scenario_name = info.get("scenario", "unknown")
        print(f"\n[Episode {ep}/{args.episodes}] scenario={scenario_name}")

        ep_steps = []
        ep_red_reward = 0.0
        ep_blue_reward = 0.0
        winner = "DRAW"

        for step in range(1, args.max_steps + 1):
            # Use internal step to access action metadata for demo output.
            red_action = red_agent.act(state)
            next_state, red_reward, red_done, _, red_info = env._red_step(red_action)
            ep_red_reward += red_reward

            print(
                f"[E{ep:02d} S{step:02d}] RED  -> {red_info['attack']:<24} "
                f"success={red_info['success']} reward={red_reward:+6.1f} "
                f"kc={red_info['kill_chain_stage']} down={len(env.interfaces_down)}"
            )

            ep_steps.append({
                "step": step,
                "actor": "red",
                "action_id": red_action,
                "action_name": RED_ACTIONS[red_action]["name"],
                "reward": red_reward,
                "success": red_info.get("success"),
                "kill_chain_stage": red_info.get("kill_chain_stage"),
                "active_attacks": red_info.get("active_attacks"),
                "interfaces_down": len(env.interfaces_down),
                "honeypot_triggered": red_info.get("honeypot_triggered"),
            })

            if red_done:
                winner = "RED"
                state = next_state
                break

            blue_action = blue_agent.act(next_state)
            next_state2, blue_reward, blue_done, _, blue_info = env._blue_step(blue_action)
            ep_blue_reward += blue_reward

            cleared = blue_info.get("cleared_attacks") or []
            cleared_str = ",".join(cleared) if cleared else "-"

            print(
                f"[E{ep:02d} S{step:02d}] BLUE -> {blue_info['defense']:<24} "
                f"success={blue_info['success']} reward={blue_reward:+6.1f} "
                f"cleared={cleared_str} down={blue_info['interfaces_down']}"
            )

            ep_steps.append({
                "step": step,
                "actor": "blue",
                "action_id": blue_action,
                "action_name": BLUE_ACTIONS[blue_action]["name"],
                "reward": blue_reward,
                "success": blue_info.get("success"),
                "kill_chain_stage": blue_info.get("kill_chain_stage"),
                "active_attacks": blue_info.get("active_attacks"),
                "interfaces_down": blue_info.get("interfaces_down"),
                "cleared_attacks": cleared,
                "chain_interrupted": blue_info.get("chain_interrupted"),
            })

            state = next_state2

            if blue_done:
                winner = "BLUE"
                break

            if args.sleep > 0:
                time.sleep(args.sleep)

        metrics = env.episode_metrics
        print(
            f"[Episode {ep}] winner={winner} red_reward={ep_red_reward:+.1f} "
            f"blue_reward={ep_blue_reward:+.1f} steps={metrics['total_steps']}"
        )

        demo_log.append({
            "episode": ep,
            "scenario": scenario_name,
            "winner": winner,
            "red_reward": ep_red_reward,
            "blue_reward": ep_blue_reward,
            "total_steps": metrics["total_steps"],
            "time_to_detect": metrics["time_to_detect"],
            "time_to_recover": metrics["time_to_recover"],
            "kill_chain_stage_reached": max(
                metrics.get("kill_chain_stage_reached", 0), env.kill_chain_stage
            ),
            "honeypot_triggered": metrics["honeypot_triggered"],
            "steps": ep_steps,
        })

    if log_path is not None:
        with open(log_path, "w") as f:
            json.dump(demo_log, f, indent=2)
        print(f"\n[+] Demo log saved to: {log_path}")


if __name__ == "__main__":
    main()
