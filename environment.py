"""
Network Environment (v2 — Full Cybersecurity Simulation)
=========================================================
Gymnasium-compatible RL environment for Red-vs-Blue cybersecurity training.

New in v2
---------
- 18 Red attack actions  (disruption / recon / persistence)
- 18 Blue defense actions (immediate / recovery / proactive)
- 80-dimensional state vector (10 interfaces × 8 metrics)
- Kill chain system with 4 stages — episode ends when Red completes all 4
- 5 scenario types (normal / already_compromised / high_traffic / partial_defense / stealth)
- Smarter reward functions with stealth bonus, honeypot, response-speed bonus
- Full episode metrics for thesis logging

Observation space : Box(80,)   — continuous vector (see state_collector.py)
Action space      : Discrete(18) — index into RED_ACTIONS or BLUE_ACTIONS
"""

import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gns3_connector  import GNS3Connector
from state_collector import StateCollector
from attack_executor import AttackExecutor
from defense_executor import DefenseExecutor
from config import (
    # Topology
    INTERFACE_ACTIONS, CORE_INTERFACES, N_INTERFACES,
    # Action catalogs
    RED_ACTIONS, BLUE_ACTIONS, ACTION_DIM, STATE_DIM,
    # Kill chain
    KILL_CHAIN_STAGE_MAP, KILL_CHAIN_RECON_THRESHOLD,
    KILL_CHAIN_COMPLETION_REWARD, KILL_CHAIN_INTERRUPT_REWARD,
    KILL_CHAIN_ENDS_EPISODE, CHAIN_INTERRUPT_BONUS,
    # Scenarios
    SCENARIO_TYPES, STEALTH_ANOMALY_THRESHOLD,
    # Rewards
    STEALTH_BONUS, HONEYPOT_PENALTY, HONEYPOT_BONUS,
    CORE_DOWN_STEP_PENALTY, WASTED_MOVE_PENALTY,
    RESPONSE_SPEED_BONUS_MAX, PROACTIVE_DETECT_BONUS,
    # Runtime
    STEP_DELAY, TELEMETRY_MODE, SYNTHETIC_SEED,
)


class NetworkEnvironment(gym.Env):
    """
    Red-vs-Blue Cybersecurity Gymnasium Environment (v2).

    mode="red"  → step() / red_step()  executes an attack.
    mode="blue" → step() / blue_step() executes a defense.
    """

    metadata = {"render_modes": []}

    # ── Construction ─────────────────────────────────────

    def __init__(self, mode: str = "red", step_delay: float = None):
        super().__init__()

        self.mode = mode
        self.step_delay = (
            step_delay if step_delay is not None
            else (0.0 if TELEMETRY_MODE == "synthetic" else STEP_DELAY)
        )

        print(f"[*] Initialising NetworkEnvironment v2 (mode={mode}) ...")

        # GNS3 + telemetry
        self.gns3      = GNS3Connector()
        self.collector = StateCollector()

        # Action executors
        self.attack_exec  = AttackExecutor(self.gns3)
        self.defense_exec = DefenseExecutor(self.gns3)

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_DIM)

        # RNG for synthetic mode
        self._rng = (
            np.random.default_rng(SYNTHETIC_SEED)
            if TELEMETRY_MODE == "synthetic" else None
        )

        # Internal state (shared between environment and executors)
        self._init_internal_state()

        print(
            f"[+] Environment ready | STATE_DIM={STATE_DIM} | "
            f"ACTION_DIM={ACTION_DIM} | mode={mode}"
        )

    # ── Internal state initialisation ────────────────────

    def _init_internal_state(self):
        """Allocate all mutable internal state arrays."""
        N = N_INTERFACES

        # 8 metric arrays (used by both synthetic and influx modes for reward calc)
        self._status         = np.ones(N,  dtype=np.float32)   # up/down
        self._traffic        = np.zeros(N, dtype=np.float32)   # log-scaled
        self._errors         = np.zeros(N, dtype=np.float32)   # normalised
        self._cpu            = np.ones(N,  dtype=np.float32) * 0.10  # 10 % idle
        self._pkt_loss       = np.zeros(N, dtype=np.float32)
        self._broadcast_ratio= np.zeros(N, dtype=np.float32)
        self._arp_size       = np.ones(N,  dtype=np.float32) * 0.05  # ~10 entries
        self._anomaly        = np.zeros(N, dtype=np.float32)

        # Link-level tracking (for GNS3 calls)
        self.interfaces_down: set = set()

        # Attack tracking
        self.active_attacks: dict = {}          # {attack_name: {action, affected, step, severity}}

        # Persistence flags
        self._backdoor_active   = False
        self._config_tampered   = False
        self._firewall_bypassed = False

        # Honeypot
        self.honeypot_active    = False
        self.honeypot_triggered = False

        # Kill chain
        self.kill_chain_stage   = 0
        self.kill_chain_counts  = {stage: 0 for stage in range(1, 5)}

        # Blue tracking
        self.attack_detected_step: int | None = None

        # Step counter
        self._step_count = 0

        # Episode metrics (reset each episode)
        self._init_episode_metrics()

    def _init_episode_metrics(self):
        """Reset per-episode thesis logging metrics."""
        self.episode_metrics = {
            "scenario":                None,
            "attack_types_used":       [],      # list of attack names used
            "defense_types_used":      [],      # list of defense names used
            "time_to_detect":          None,    # steps until Blue detected attack
            "time_to_recover":         None,    # steps until full recovery
            "availability_steps":      0,       # steps with zero interfaces down
            "total_steps":             0,
            "false_positives":         0,       # Blue acted when nothing was active
            "attack_success_count":    {},      # {attack_name: successes}
            "defense_success_count":   {},      # {defense_name: successes}
            "kill_chain_stage_reached":0,
            "honeypot_triggered":      False,
            "chain_interrupted":       False,
        }

    # ── Gymnasium API ─────────────────────────────────────

    def reset(self, seed=None, options=None, scenario: str = None):
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        scenario : str | None
            One of SCENARIO_TYPES, or None to pick randomly.
        """
        super().reset(seed=seed)

        # Restore all links to GNS3
        print("[*] Resetting environment — restoring all links ...")
        for node, port in INTERFACE_ACTIONS:
            self.gns3.restore_interface(node, port)
        time.sleep(self.step_delay + 0.5)

        # Re-initialise all internal state
        self._init_internal_state()

        # Pick and apply scenario
        chosen = scenario or random.choice(SCENARIO_TYPES)
        self.episode_metrics["scenario"] = chosen
        self._apply_scenario(chosen)

        print(f"[+] Reset complete | scenario={chosen}")
        obs = self._build_state()
        return obs, {"scenario": chosen}

    def step(self, action: int):
        """Route to red_step or blue_step based on current mode."""
        if self.mode == "red":
            return self._red_step(action)
        return self._blue_step(action)

    # ── Red step ─────────────────────────────────────────

    def _red_step(self, index: int):
        """Execute one Red-agent attack action."""
        if index >= ACTION_DIM:
            obs = self._build_state()
            return obs, -1.0, False, False, {"valid": False}

        action = RED_ACTIONS[index]
        self._step_count += 1
        self.episode_metrics["total_steps"] += 1

        # ── Execute attack ──────────────────────────────
        success, affected, severity = self.attack_exec.execute(action, self)

        # ── Update kill chain ───────────────────────────
        if success:
            self._update_kill_chain(action)
            self.episode_metrics["attack_types_used"].append(action["name"])
            self.episode_metrics["attack_success_count"].setdefault(action["name"], 0)
            self.episode_metrics["attack_success_count"][action["name"]] += 1

        # ── Check honeypot ──────────────────────────────
        if self.honeypot_triggered:
            self.episode_metrics["honeypot_triggered"] = True

        # ── Availability tracking ───────────────────────
        if len(self.interfaces_down) == 0 and not self.active_attacks:
            self.episode_metrics["availability_steps"] += 1

        # ── Compute reward ──────────────────────────────
        reward = self._compute_red_reward(action, success, affected)

        # ── Check kill chain completion ─────────────────
        terminated = False
        if KILL_CHAIN_ENDS_EPISODE and self.kill_chain_stage >= 4:
            reward     += KILL_CHAIN_COMPLETION_REWARD
            terminated  = True
            self.episode_metrics["kill_chain_stage_reached"] = 4
            print(f"[!] KILL CHAIN COMPLETE — Red wins! (+{KILL_CHAIN_COMPLETION_REWARD})")

        # ── Synthetic state decay ───────────────────────
        self._decay_synthetic_state()

        obs  = self._build_state()
        info = {
            "attack":             action["name"],
            "success":            success,
            "affected_ifaces":    affected,
            "kill_chain_stage":   self.kill_chain_stage,
            "active_attacks":     list(self.active_attacks.keys()),
            "honeypot_triggered": self.honeypot_triggered,
        }
        return obs, reward, terminated, False, info

    # ── Blue step ─────────────────────────────────────────

    def _blue_step(self, index: int):
        """Execute one Blue-agent defense action."""
        if index >= ACTION_DIM:
            obs = self._build_state()
            return obs, -1.0, False, False, {"valid": False}

        action = BLUE_ACTIONS[index]
        self._step_count += 1
        self.episode_metrics["total_steps"] += 1

        attacks_before = set(self.active_attacks.keys())
        was_down       = len(self.interfaces_down) > 0

        # ── Execute defense ─────────────────────────────
        success, cleared = self.defense_exec.execute(action, self)

        # ── Track detection timing ──────────────────────
        if success and self.attack_detected_step is None and attacks_before:
            self.attack_detected_step = self._step_count

        # ── Track recovery ──────────────────────────────
        if len(self.interfaces_down) == 0 and not self.active_attacks:
            if self.episode_metrics["time_to_recover"] is None and was_down:
                self.episode_metrics["time_to_recover"] = self._step_count

        # ── Log false positives ─────────────────────────
        if not attacks_before and not was_down and not self.honeypot_triggered:
            self.episode_metrics["false_positives"] += 1

        # ── Log defense usage ───────────────────────────
        if success:
            self.episode_metrics["defense_types_used"].append(action["name"])
            self.episode_metrics["defense_success_count"].setdefault(action["name"], 0)
            self.episode_metrics["defense_success_count"][action["name"]] += 1

        # ── Availability tracking ───────────────────────
        if len(self.interfaces_down) == 0 and not self.active_attacks:
            self.episode_metrics["availability_steps"] += 1

        # ── Compute reward ──────────────────────────────
        reward = self._compute_blue_reward(action, success, attacks_before, cleared)

        # ── Check if Blue interrupted the kill chain ────
        chain_interrupted = False
        if self.kill_chain_stage < 4 and attacks_before and not self.active_attacks:
            chain_interrupted = True
            self.episode_metrics["chain_interrupted"] = True
            reward += CHAIN_INTERRUPT_BONUS
            print(f"[!] Kill chain interrupted at stage {self.kill_chain_stage}! "
                  f"(+{CHAIN_INTERRUPT_BONUS})")

        # ── Termination condition ───────────────────────
        # Blue wins if all interfaces are up and no attacks are active
        terminated = (len(self.interfaces_down) == 0
                      and not self.active_attacks
                      and not self._config_tampered
                      and not self._backdoor_active)

        if terminated:
            self.episode_metrics["time_to_recover"] = (
                self.episode_metrics["time_to_recover"] or self._step_count
            )

        # ── Synthetic state decay ───────────────────────
        self._decay_synthetic_state()

        obs  = self._build_state()
        info = {
            "defense":           action["name"],
            "success":           success,
            "cleared_attacks":   cleared,
            "interfaces_down":   len(self.interfaces_down),
            "active_attacks":    list(self.active_attacks.keys()),
            "kill_chain_stage":  self.kill_chain_stage,
            "chain_interrupted": chain_interrupted,
        }
        return obs, reward, terminated, False, info

    # ── Old-API wrappers (used by train.py) ───────────────

    def red_step(self, action: int):
        """Returns (obs, reward, done) — backward-compatible API."""
        obs, reward, terminated, truncated, info = self._red_step(action)
        return obs, reward, terminated or truncated

    def blue_step(self, action: int):
        """Returns (obs, reward, done) — backward-compatible API."""
        obs, reward, terminated, truncated, info = self._blue_step(action)
        return obs, reward, terminated or truncated

    # ── Reward computation ────────────────────────────────

    def _compute_red_reward(self, action: dict, success: bool, affected: list) -> float:
        if not success:
            return -1.0

        base = action["base_reward"]

        # Core multiplier: core targets pay extra
        if action["target"] == "core":
            reward = base * action.get("core_multiplier", 1.0)
        else:
            # Scale by number of affected interfaces (more impact = more reward)
            impact  = max(len(affected), 1) / N_INTERFACES
            reward  = base * (0.7 + 0.3 * impact)

        # Stealth bonus: low anomaly score = undetected attack
        avg_anomaly = float(np.mean(self._anomaly))
        if avg_anomaly < STEALTH_ANOMALY_THRESHOLD:
            reward += STEALTH_BONUS * action.get("stealth", 0.5)

        # Kill chain stage bonus: later stages reward more
        if self.kill_chain_stage > 0:
            reward += self.kill_chain_stage * 2.0

        # Honeypot penalty: got caught
        if self.honeypot_triggered:
            reward -= HONEYPOT_PENALTY
            self.honeypot_triggered = False  # Reset trap after penalty applied

        return float(reward)

    def _compute_blue_reward(
        self,
        action: dict,
        success: bool,
        attacks_before: set,
        cleared: list,
    ) -> float:

        # Penalty for acting when no attack matched
        if not attacks_before and not self.honeypot_triggered:
            if not success:
                return -WASTED_MOVE_PENALTY

        if not success:
            return -WASTED_MOVE_PENALTY

        base = action["base_reward"]

        # Core down penalty: extra urgency while core infra is under attack
        if self._core_is_down():
            base -= CORE_DOWN_STEP_PENALTY

        # Successful counter of an active attack
        if cleared:
            # Full recovery bonus
            all_clear = (len(self.interfaces_down) == 0
                         and not self.active_attacks
                         and not self._backdoor_active
                         and not self._config_tampered)
            if all_clear:
                base *= 2.0

            # Response speed bonus: fewer steps since attack = bigger bonus
            if attacks_before:
                # Find when the first attack in 'cleared' was launched
                earliest_attack_step = self._step_count  # worst case: same step
                for atk_name, atk_info in list(self.active_attacks.items()) + [
                    (n, {}) for n in cleared
                ]:
                    step = atk_info.get("step", self._step_count) if isinstance(atk_info, dict) else self._step_count
                    if step < earliest_attack_step:
                        earliest_attack_step = step
                response_steps = self._step_count - earliest_attack_step
                if response_steps <= 1:
                    base += RESPONSE_SPEED_BONUS_MAX
                elif response_steps <= 3:
                    base += RESPONSE_SPEED_BONUS_MAX * 0.5

        # Proactive detection bonus: Blue detected before damage (attacks_before existed)
        if action["category"] == "proactive" and attacks_before:
            base += PROACTIVE_DETECT_BONUS * 0.5

        # Honeypot bonus: successfully trapped Red
        if self.honeypot_triggered and action["name"] == "Honeypot Deployment":
            base += HONEYPOT_BONUS
            self.episode_metrics["honeypot_triggered"] = True

        return float(max(base, -WASTED_MOVE_PENALTY))

    # ── Kill chain ────────────────────────────────────────

    def _update_kill_chain(self, action: dict):
        """Advance kill chain stage based on action category and name."""
        action_name = action["name"]

        for stage, (_, action_names) in KILL_CHAIN_STAGE_MAP.items():
            if action_name in action_names and stage > self.kill_chain_stage:
                # Stage 1 requires KILL_CHAIN_RECON_THRESHOLD recon actions
                if stage == 1:
                    self.kill_chain_counts[1] = self.kill_chain_counts.get(1, 0) + 1
                    if self.kill_chain_counts[1] >= KILL_CHAIN_RECON_THRESHOLD:
                        self.kill_chain_stage = 1
                        print(f"[*] Kill chain Stage 1 — RECON complete")
                # Higher stages require previous stage to be complete
                elif stage == self.kill_chain_stage + 1:
                    self.kill_chain_stage = stage
                    print(f"[*] Kill chain Stage {stage} — "
                          f"{list(KILL_CHAIN_STAGE_MAP[stage])[0].upper()} complete")
                break

        self.episode_metrics["kill_chain_stage_reached"] = max(
            self.episode_metrics["kill_chain_stage_reached"],
            self.kill_chain_stage,
        )

    # ── Scenario initialisation ───────────────────────────

    def _apply_scenario(self, scenario: str):
        """Configure episode starting conditions based on selected scenario."""
        if scenario == "normal":
            # Default: all links up, clean state
            pass

        elif scenario == "already_compromised":
            # One random non-core interface already down
            idx  = random.randint(0, len(INTERFACE_ACTIONS) - 3)   # avoid core
            node, port = INTERFACE_ACTIONS[idx]
            self.gns3.shutdown_interface(node, port)
            self.interfaces_down.add((node, port))
            self._status[idx % N_INTERFACES] = 0.0
            print(f"[scenario] already_compromised: {node} port {port} is down")

        elif scenario == "high_traffic":
            # Simulate heavy pre-existing load
            rng = self._rng or np.random.default_rng()
            self._traffic      = np.log1p(rng.uniform(5e4, 5e5, N_INTERFACES)).astype(np.float32)
            self._cpu          = np.clip(
                rng.uniform(0.5, 0.85, N_INTERFACES).astype(np.float32), 0, 1
            )
            print("[scenario] high_traffic: elevated traffic and CPU pre-loaded")

        elif scenario == "partial_defense":
            # Blue starts with honeypot already deployed
            self.honeypot_active = True
            print("[scenario] partial_defense: honeypot pre-deployed")

        elif scenario == "stealth":
            # Anomaly threshold is lowered — Blue is more vigilant
            # Red must stay VERY quiet (actions with stealth < 0.6 are harder)
            self._anomaly[:] = 0.05   # Near-zero baseline — any attack stands out
            print("[scenario] stealth: near-zero anomaly baseline — Red must be silent")

    # ── State building ────────────────────────────────────

    def _build_state(self) -> np.ndarray:
        """Build the 80-dim observation vector."""
        if TELEMETRY_MODE == "synthetic":
            return self._synthetic_state_vector()
        # Influx mode: pull live data but also merge internal attack-effect arrays
        live = self.collector.get_state_vector()
        return self._merge_influx_with_internal(live)

    def _synthetic_state_vector(self) -> np.ndarray:
        """
        Build the 80-dim state vector from internal synthetic arrays.
        Adds background noise to simulate realistic telemetry fluctuation.
        """
        rng = self._rng or np.random.default_rng()

        # Background noise — small perturbations on all metrics
        noise = rng.normal(0, 0.02, N_INTERFACES).astype(np.float32)

        traffic_bg = np.log1p(
            rng.uniform(1e3, 1e5, N_INTERFACES)
        ).astype(np.float32) * self._status  # Zero traffic on down interfaces

        errors_bg = np.clip(
            rng.uniform(0, 0.5, N_INTERFACES).astype(np.float32), 0, 10
        )

        # Merge background with attack-induced state
        traffic = np.clip(traffic_bg + self._traffic,       0, None)
        errors  = np.clip(errors_bg  + self._errors,        0, 10)
        cpu     = np.clip(self._cpu  + noise,               0, 1)
        pkt     = np.clip(self._pkt_loss + abs(noise),      0, 1)
        bcast   = np.clip(self._broadcast_ratio + abs(noise * 0.5), 0, 1)
        arp     = np.clip(self._arp_size + abs(noise * 0.3), 0, 1)
        anomaly = np.clip(self._anomaly + abs(noise * 0.1), 0, 1)

        vec = np.concatenate([
            self._status,   # [0:10]
            traffic,        # [10:20]
            errors,         # [20:30]
            cpu,            # [30:40]
            pkt,            # [40:50]
            bcast,          # [50:60]
            arp,            # [60:70]
            anomaly,        # [70:80]
        ])
        return vec.astype(np.float32)

    def _merge_influx_with_internal(self, live: np.ndarray) -> np.ndarray:
        """
        In influx mode, overlay internal attack perturbations on live telemetry.
        This ensures the reward system sees consistent attack effects even when
        InfluxDB doesn't fully reflect the simulated attacks.
        """
        N = N_INTERFACES
        merged = live.copy()

        # Status: use GNS3 link state as ground truth
        for i, (node, port) in enumerate(INTERFACE_ACTIONS):
            if i < N and (node, port) in self.interfaces_down:
                merged[i] = 0.0

        # Overlay internal perturbations on top of live metrics
        merged[2*N:3*N] = np.clip(live[2*N:3*N] + self._errors,        0, 10)  # errors
        merged[3*N:4*N] = np.clip(live[3*N:4*N] + self._cpu - 0.1,     0, 1)   # cpu
        merged[4*N:5*N] = np.clip(live[4*N:5*N] + self._pkt_loss,      0, 1)   # pkt_loss
        merged[5*N:6*N] = np.clip(live[5*N:6*N] + self._broadcast_ratio, 0, 1) # bcast
        merged[6*N:7*N] = np.clip(live[6*N:7*N] + self._arp_size - 0.05, 0, 1) # arp
        merged[7*N:8*N] = np.clip(live[7*N:8*N] + self._anomaly,       0, 1)   # anomaly

        return merged.astype(np.float32)

    # ── Helpers ──────────────────────────────────────────

    def _core_is_down(self) -> bool:
        """True if any core interface is currently down or under core attack."""
        for (node, port) in CORE_INTERFACES:
            if (node, port) in self.interfaces_down:
                return True
        core_attack_names = {"Shutdown Core Link", "DDoS Core Router"}
        return bool(self.active_attacks.keys() & core_attack_names)

    def _decay_synthetic_state(self):
        """Gradually decay attack-induced state perturbations (attacks fade over time)."""
        DECAY = 0.08  # per-step decay factor
        self._errors          = np.clip(self._errors          - DECAY * 0.5, 0, 10)
        self._pkt_loss        = np.clip(self._pkt_loss        - DECAY,        0, 1)
        self._broadcast_ratio = np.clip(self._broadcast_ratio - DECAY,        0, 1)
        self._arp_size        = np.clip(self._arp_size        - DECAY * 0.3,  0, 1)
        self._cpu             = np.clip(self._cpu             - DECAY * 0.5,  0, 1)
        self._anomaly         = np.clip(self._anomaly         - DECAY * 0.15, 0, 1)
        # Traffic recovers toward zero
        self._traffic         = np.clip(self._traffic         - DECAY * 2.0,  0, None)


# ==================== TEST BLOCK ====================
if __name__ == "__main__":
    import time

    print("\n" + "="*60)
    print("  Environment v2 — Smoke Test (synthetic mode)")
    print("="*60)

    # Force synthetic mode for testing
    import config as _cfg
    _cfg.TELEMETRY_MODE = "synthetic"
    from importlib import reload
    import config
    reload(config)

    env = NetworkEnvironment(mode="red")
    obs, info = env.reset(scenario="normal")
    print(f"\nObs shape  : {obs.shape}")
    print(f"Scenario   : {info['scenario']}")
    print(f"Obs[:10]   : {obs[:10]}")   # status

    print("\n--- Red: Port Scan (recon stage 1) ---")
    obs, r, term, trunc, info = env.step(7)   # Port Scan
    print(f"Reward={r:.2f} | KillChain={info['kill_chain_stage']} | {info}")

    print("\n--- Red: Topology Mapping (recon stage 1 complete) ---")
    obs, r, term, trunc, info = env.step(8)   # Topology Mapping
    print(f"Reward={r:.2f} | KillChain={info['kill_chain_stage']}")

    print("\n--- Red: ARP Poisoning (intrusion stage 2) ---")
    obs, r, term, trunc, info = env.step(2)   # ARP Poisoning
    print(f"Reward={r:.2f} | KillChain={info['kill_chain_stage']}")

    # Switch to Blue
    env.mode = "blue"
    print("\n--- Blue: ARP Inspection Enable ---")
    obs, r, term, trunc, info = env.step(15)  # ARP Inspection Enable
    print(f"Reward={r:.2f} | Cleared={info['cleared_attacks']} | Done={term}")

    print("\n--- Blue: Honeypot Deployment ---")
    obs, r, term, trunc, info = env.step(9)   # Honeypot Deployment
    print(f"Reward={r:.2f} | Info={info}")

    print("\n✓ Smoke test complete.")
