"""
Red-vs-Blue Self-Play Training Loop (v2)
=========================================
Trains Red (attacker) and Blue (defender) DQN agents against each other
in the GNS3 Network Digital Twin environment.

New in v2
---------
- Full thesis metrics logged per episode (attack types, detection time,
  recovery time, availability %, false positives, kill chain stage, etc.)
- Per-attack-type and per-defense-type success rate tracking
- Scenario type logged each episode
- Honeypot trigger count logged
- TensorBoard extended with new scalars
"""

import os
import csv
import json
import time
from datetime import datetime
from collections import defaultdict

from environment import NetworkEnvironment
from red_agent   import RedAgent
from blue_agent  import BlueAgent
from config      import (
    EPISODES, MAX_STEPS_PER_EP, SAVE_INTERVAL,
    LOG_DIR, MODEL_DIR,
    RED_ACTIONS, BLUE_ACTIONS,
)

# ── Optional TensorBoard ──────────────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False


# ── Setup ─────────────────────────────────────────────────

def setup_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)


def _safe(val, fmt=".2f"):
    """Format a numeric value; return '—' if None."""
    return format(val, fmt) if val is not None else "—"


# ── Training loop ─────────────────────────────────────────

def train():
    setup_dirs()

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path   = os.path.join(LOG_DIR, f"training_{timestamp}.csv")
    stats_path = os.path.join(LOG_DIR, f"attack_defense_stats_{timestamp}.json")

    # ── CSV columns (thesis-ready) ─────────────────────────
    CSV_COLUMNS = [
        # Episode basics
        "episode", "scenario", "epsilon",
        # Rewards
        "red_reward", "blue_reward",
        # Losses
        "red_loss_avg", "blue_loss_avg",
        # Win tracking
        "red_wins", "blue_wins",
        # Kill chain
        "kill_chain_stage_reached", "chain_interrupted",
        # Time metrics
        "time_to_detect", "time_to_recover",
        # Network health
        "network_availability_pct",
        # Quality metrics
        "false_positive_rate",
        "honeypot_triggered",
        # Attack / defense summaries (serialised as JSON strings)
        "attack_types_used", "defense_types_used",
        "attack_success_counts", "defense_success_counts",
    ]

    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    csv_writer.writeheader()

    tb_writer = None
    if TB_AVAILABLE:
        tb_dir    = os.path.join(LOG_DIR, f"tb_{timestamp}")
        tb_writer = SummaryWriter(log_dir=tb_dir)
        print(f"[+] TensorBoard logs → {tb_dir}")

    print(f"[+] CSV logs  → {csv_path}")
    print(f"[+] Stats     → {stats_path}")
    print(f"[+] Models    → {MODEL_DIR}/")

    # ── Agents & Environment ───────────────────────────────
    env        = NetworkEnvironment()
    red_agent  = RedAgent()
    blue_agent = BlueAgent()

    # ── Cumulative stats ───────────────────────────────────
    total_red_wins   = 0
    total_blue_wins  = 0
    all_red_rewards  = []
    all_blue_rewards = []

    # Per-attack / per-defense running success rates
    atk_stats  = defaultdict(lambda: {"attempts": 0, "successes": 0})
    def_stats  = defaultdict(lambda: {"attempts": 0, "successes": 0})

    print(f"\n{'='*65}")
    print(f"  Training v2 | {EPISODES} episodes × {MAX_STEPS_PER_EP} steps")
    print(f"  STATE_DIM=80 | ACTION_DIM=18 | 5 scenarios | kill-chain")
    print(f"{'='*65}\n")

    t_start = time.time()

    for episode in range(1, EPISODES + 1):
        state, ep_info = env.reset()
        scenario       = ep_info.get("scenario", "normal")

        ep_red_reward  = 0.0
        ep_blue_reward = 0.0
        red_losses     = []
        blue_losses    = []

        for step in range(MAX_STEPS_PER_EP):

            # ── Red attacks ────────────────────────────────
            red_action = red_agent.act(state)
            next_state, red_reward, red_done = env.red_step(red_action)
            red_agent.remember(state, red_action, red_reward, next_state)
            loss = red_agent.train()
            if loss is not None:
                red_losses.append(loss)
            ep_red_reward += red_reward

            # Track per-attack-type stats
            atk_name = RED_ACTIONS[red_action]["name"]
            atk_stats[atk_name]["attempts"] += 1
            if red_reward > 0:
                atk_stats[atk_name]["successes"] += 1

            if red_done:
                break   # Kill chain completed — episode over

            # ── Blue defends ───────────────────────────────
            blue_action = blue_agent.act(next_state)
            next_state2, blue_reward, blue_done = env.blue_step(blue_action)
            blue_agent.remember(next_state, blue_action, blue_reward, next_state2)
            loss = blue_agent.train()
            if loss is not None:
                blue_losses.append(loss)
            ep_blue_reward += blue_reward

            # Track per-defense-type stats
            def_name = BLUE_ACTIONS[blue_action]["name"]
            def_stats[def_name]["attempts"] += 1
            if blue_reward > 0:
                def_stats[def_name]["successes"] += 1

            state = next_state2

            if blue_done:
                break

        # ── Episode metrics ────────────────────────────────
        metrics = env.episode_metrics
        total_steps = max(metrics["total_steps"], 1)

        avail_pct = (metrics["availability_steps"] / total_steps) * 100.0
        fp_rate   = metrics["false_positives"] / total_steps

        red_won  = ep_red_reward > ep_blue_reward
        blue_won = ep_blue_reward > ep_red_reward
        total_red_wins  += int(red_won)
        total_blue_wins += int(blue_won)
        all_red_rewards.append(ep_red_reward)
        all_blue_rewards.append(ep_blue_reward)

        avg_red_loss  = sum(red_losses)  / len(red_losses)  if red_losses  else 0.0
        avg_blue_loss = sum(blue_losses) / len(blue_losses) if blue_losses else 0.0

        # ── Console output ─────────────────────────────────
        winner = "RED " if red_won else ("BLUE" if blue_won else "DRAW")
        print(
            f"Ep {episode:4d}/{EPISODES} | "
            f"{scenario:20s} | "
            f"Red: {ep_red_reward:+7.1f} | Blue: {ep_blue_reward:+7.1f} | "
            f"ε={red_agent.epsilon:.3f} | KC={metrics['kill_chain_stage_reached']} | "
            f"Avail={avail_pct:4.0f}% | {winner} "
            f"R:{total_red_wins} B:{total_blue_wins}"
        )

        # ── CSV log ────────────────────────────────────────
        row = {
            "episode":                   episode,
            "scenario":                  scenario,
            "epsilon":                   f"{red_agent.epsilon:.4f}",
            "red_reward":                f"{ep_red_reward:.2f}",
            "blue_reward":               f"{ep_blue_reward:.2f}",
            "red_loss_avg":              f"{avg_red_loss:.6f}",
            "blue_loss_avg":             f"{avg_blue_loss:.6f}",
            "red_wins":                  total_red_wins,
            "blue_wins":                 total_blue_wins,
            "kill_chain_stage_reached":  metrics["kill_chain_stage_reached"],
            "chain_interrupted":         int(metrics["chain_interrupted"]),
            "time_to_detect":            _safe(metrics["time_to_detect"], "d"),
            "time_to_recover":           _safe(metrics["time_to_recover"], "d"),
            "network_availability_pct":  f"{avail_pct:.1f}",
            "false_positive_rate":       f"{fp_rate:.4f}",
            "honeypot_triggered":        int(metrics["honeypot_triggered"]),
            "attack_types_used":         json.dumps(metrics["attack_types_used"]),
            "defense_types_used":        json.dumps(metrics["defense_types_used"]),
            "attack_success_counts":     json.dumps(metrics["attack_success_count"]),
            "defense_success_counts":    json.dumps(metrics["defense_success_count"]),
        }
        csv_writer.writerow(row)
        csv_file.flush()

        # ── TensorBoard ────────────────────────────────────
        if tb_writer:
            tb_writer.add_scalars("Reward",
                {"Red": ep_red_reward, "Blue": ep_blue_reward}, episode)
            tb_writer.add_scalar("Epsilon", red_agent.epsilon, episode)
            tb_writer.add_scalar("KillChain/StageReached",
                metrics["kill_chain_stage_reached"], episode)
            tb_writer.add_scalar("Network/Availability_pct",   avail_pct, episode)
            tb_writer.add_scalar("Network/FalsePositiveRate",  fp_rate,   episode)
            if metrics["time_to_detect"] is not None:
                tb_writer.add_scalar("Blue/TimeToDetect",
                    metrics["time_to_detect"], episode)
            if metrics["time_to_recover"] is not None:
                tb_writer.add_scalar("Blue/TimeToRecover",
                    metrics["time_to_recover"], episode)
            tb_writer.add_scalars("Wins",
                {"Red": total_red_wins, "Blue": total_blue_wins}, episode)
            if red_losses:
                tb_writer.add_scalar("Loss/Red",  avg_red_loss,  episode)
            if blue_losses:
                tb_writer.add_scalar("Loss/Blue", avg_blue_loss, episode)

        # ── Save checkpoints ───────────────────────────────
        if episode % SAVE_INTERVAL == 0:
            red_agent.save(os.path.join(MODEL_DIR, "red_model.pth"))
            blue_agent.save(os.path.join(MODEL_DIR, "blue_model.pth"))
            _save_action_stats(stats_path, atk_stats, def_stats)
            print(f"      → Models + stats saved (episode {episode})")

    # ── Final save ─────────────────────────────────────────
    red_agent.save(os.path.join(MODEL_DIR, "red_model.pth"))
    blue_agent.save(os.path.join(MODEL_DIR, "blue_model.pth"))
    _save_action_stats(stats_path, atk_stats, def_stats)

    elapsed = time.time() - t_start
    csv_file.close()
    if tb_writer:
        tb_writer.close()

    # ── Final summary ──────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Training Complete (v2)")
    print(f"{'='*65}")
    print(f"  Episodes        : {EPISODES}")
    print(f"  Time            : {elapsed/60:.1f} min")
    print(f"  Red  wins       : {total_red_wins}  ({100*total_red_wins/EPISODES:.1f}%)")
    print(f"  Blue wins       : {total_blue_wins} ({100*total_blue_wins/EPISODES:.1f}%)")
    print(f"  Avg Red reward  : {sum(all_red_rewards)/len(all_red_rewards):+.2f}")
    print(f"  Avg Blue reward : {sum(all_blue_rewards)/len(all_blue_rewards):+.2f}")
    print(f"  Final epsilon   : {red_agent.epsilon:.4f}")
    print(f"  CSV logs        : {csv_path}")
    print(f"  Action stats    : {stats_path}")
    print(f"  Models          : {MODEL_DIR}/")
    print(f"{'='*65}\n")

    # Print top-3 most used attacks and defenses
    _print_top_actions(atk_stats, label="attack")
    _print_top_actions(def_stats, label="defense")


# ── Helpers ───────────────────────────────────────────────

def _save_action_stats(path: str, atk_stats: dict, def_stats: dict):
    """Persist per-action success statistics as JSON."""
    data = {
        "attacks":  {k: dict(v) for k, v in atk_stats.items()},
        "defenses": {k: dict(v) for k, v in def_stats.items()},
    }
    # Compute success rates
    for category in data.values():
        for name, s in category.items():
            a = s["attempts"]
            s["success_rate"] = round(s["successes"] / a, 4) if a > 0 else 0.0

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _print_top_actions(stats: dict, label: str, top_n: int = 5):
    """Print the top-N most-used actions and their success rates."""
    sorted_actions = sorted(
        stats.items(),
        key=lambda x: x[1]["attempts"],
        reverse=True,
    )[:top_n]
    print(f"\n  Top {top_n} {label} actions:")
    for name, s in sorted_actions:
        rate = s["successes"] / max(s["attempts"], 1) * 100
        print(f"    {name:30s} | attempts={s['attempts']:4d} | "
              f"success={s['successes']:4d} ({rate:.0f}%)")


if __name__ == "__main__":
    train()
