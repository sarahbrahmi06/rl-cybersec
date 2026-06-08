#!/usr/bin/env python3
"""
graphs.py - Generate thesis graphs from training or evaluation CSV logs
======================================================================
Creates six PNG charts from a CSV file:
1) Episode rewards (Red vs Blue)
2) Cumulative win rate
3) Epsilon decay
4) Network availability
5) Kill-chain stage reached
6) Detection and recovery time
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def find_latest_csv(log_dir: str, prefix: str) -> Path:
    """Find the most recent CSV matching prefix in a directory."""
    candidates = sorted(
        Path(log_dir).glob(f"{prefix}*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found in {log_dir} with prefix '{prefix}'."
        )
    return candidates[0]


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric, tolerating placeholders."""
    cleaned = series.replace({"—": np.nan, "NA": np.nan, "": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def plot_rewards(df: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["episode"], df["red_reward"], label="Red", color="crimson")
    ax.plot(df["episode"], df["blue_reward"], label="Blue", color="royalblue")
    ax.set_title("Episode Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    fig.tight_layout()

    path = out_dir / "01_rewards.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_win_rates(df: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = df["episode"].replace(0, np.nan)
    red_rate = (df["red_wins"] / episodes) * 100.0
    blue_rate = (df["blue_wins"] / episodes) * 100.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["episode"], red_rate, label="Red", color="crimson")
    ax.plot(df["episode"], blue_rate, label="Blue", color="royalblue")
    ax.set_title("Cumulative Win Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win rate (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()

    path = out_dir / "02_win_rate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_epsilon(df: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["episode"], df["epsilon"], color="black")
    ax.set_title("Epsilon Decay")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    fig.tight_layout()

    path = out_dir / "03_epsilon_decay.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_availability(df: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        df["episode"],
        df["network_availability_pct"],
        color="seagreen",
    )
    ax.set_title("Network Availability")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Availability (%)")
    ax.set_ylim(0, 100)
    fig.tight_layout()

    path = out_dir / "04_availability.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_kill_chain(df: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["episode"], df["kill_chain_stage_reached"], color="purple")
    ax.set_title("Kill Chain Stage Reached")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Stage")
    ax.set_ylim(0, 4)
    fig.tight_layout()

    path = out_dir / "05_kill_chain_stage.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_detection_recovery(df: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["episode"], df["time_to_detect"], label="Detect", color="orange")
    ax.plot(df["episode"], df["time_to_recover"], label="Recover", color="slateblue")
    ax.set_title("Detection and Recovery Time")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend()
    fig.tight_layout()

    path = out_dir / "06_detect_recover.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate thesis graphs from training/evaluation CSV logs."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="CSV file to use (default: latest training_*.csv in logs/).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to search when --input is not provided (default: logs).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="training_",
        help="CSV prefix to search for (default: training_).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="graphs",
        help="Output directory for PNG graphs (default: graphs/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input is None:
        csv_path = find_latest_csv(args.log_dir, args.prefix)
    else:
        csv_path = Path(args.input)
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    df["episode"] = coerce_numeric(df["episode"])
    df["red_reward"] = coerce_numeric(df["red_reward"])
    df["blue_reward"] = coerce_numeric(df["blue_reward"])
    df["red_wins"] = coerce_numeric(df["red_wins"])
    df["blue_wins"] = coerce_numeric(df["blue_wins"])
    df["epsilon"] = coerce_numeric(df["epsilon"])
    df["network_availability_pct"] = coerce_numeric(df["network_availability_pct"])
    df["kill_chain_stage_reached"] = coerce_numeric(df["kill_chain_stage_reached"])
    df["time_to_detect"] = coerce_numeric(df["time_to_detect"])
    df["time_to_recover"] = coerce_numeric(df["time_to_recover"])

    df = df.dropna(subset=["episode"]).copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        plot_rewards(df, out_dir),
        plot_win_rates(df, out_dir),
        plot_epsilon(df, out_dir),
        plot_availability(df, out_dir),
        plot_kill_chain(df, out_dir),
        plot_detection_recovery(df, out_dir),
    ]

    print(f"[+] CSV source: {csv_path}")
    print(f"[+] Graphs saved to: {out_dir}")
    for path in outputs:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
