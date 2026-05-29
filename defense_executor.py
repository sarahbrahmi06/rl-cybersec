"""
Defense Executor
================
Simulates the effect of every Blue-agent defense action on the environment.

Design
------
- Each method corresponds to one entry in BLUE_ACTIONS from config.py.
- Returns a tuple: (success: bool, attacks_cleared: list[str])
- An action succeeds if it directly counters at least one currently-active attack.
- On success it removes the matching active attacks from env.active_attacks
  and reverses or dampens the state perturbations they caused.
"""

import random
import numpy as np

from config import (
    INTERFACE_ACTIONS, CORE_INTERFACES,
    N_INTERFACES, BLUE_ACTIONS,
    HONEYPOT_BONUS,
)


class DefenseExecutor:
    """Executes Blue-agent defenses and updates environment internal state."""

    def __init__(self, gns3_connector):
        self.gns3 = gns3_connector

        # Dispatch table: action name → handler method
        self._dispatch = {
            "Restore Link":           self._restore_link,
            "Port Isolation":         self._port_isolation,
            "MAC Blacklisting":       self._mac_blacklisting,
            "Rate Limiting":          self._rate_limiting,
            "ICMP Blocking":          self._icmp_blocking,
            "Traffic Rerouting":      self._traffic_rerouting,
            "VLAN Reassignment":      self._vlan_reassignment,
            "Config Restoration":     self._config_restoration,
            "Failover Activation":    self._failover_activation,
            "Honeypot Deployment":    self._honeypot_deployment,
            "Anomaly Detection Alert":"_anomaly_detection_alert",
            "Dynamic ACL":            self._dynamic_acl,
            "Traffic Baselining":     self._traffic_baselining,
            "Firewall Rule Update":   self._firewall_rule_update,
            "STP Guard Activation":   self._stp_guard_activation,
            "ARP Inspection Enable":  self._arp_inspection_enable,
            "IDS Alert Response":     self._ids_alert_response,
            "Full Network Lockdown":  self._full_network_lockdown,
        }
        # Fix the string bug above
        self._dispatch["Anomaly Detection Alert"] = self._anomaly_detection_alert

    # ── Public API ───────────────────────────────────────

    def execute(self, action_dict: dict, env) -> tuple:
        """
        Execute the given defense action against the environment.

        Parameters
        ----------
        action_dict : dict
            An entry from BLUE_ACTIONS.
        env : NetworkEnvironment
            The live environment (state is read and mutated here).

        Returns
        -------
        (success, attacks_cleared)
            success        : bool  — True if the action had a useful effect
            attacks_cleared: list  — names of Red attacks that were neutralised
        """
        name    = action_dict["name"]
        handler = self._dispatch.get(name, self._default)
        return handler(action_dict, env)

    # ── Recovery actions ─────────────────────────────────

    def _restore_link(self, action, env):
        """Restore a suspended GNS3 link (real API call)."""
        affected = self._affected_ifaces(env, action["counters"])
        if not affected and not env.interfaces_down:
            return False, []

        # Restore affected links (or any downed link if none mapped)
        if affected:
            self._restore_ifaces(env, affected)
        else:
            # Fallback: restore one random downed link
            node, port = random.choice(list(env.interfaces_down))
            env.gns3.restore_interface(node, port)
            env.interfaces_down.discard((node, port))
            idx = next(
                (i for i, (n, p) in enumerate(INTERFACE_ACTIONS) if n == node and p == port),
                None,
            )
            if idx is not None and idx < N_INTERFACES:
                env._status[idx] = 1.0
                env._errors[idx] = max(0.0, env._errors[idx] - 2.0)

        cleared = self._clear_attacks(env, action["counters"])
        return True, cleared

    def _traffic_rerouting(self, action, env):
        """Reroute traffic through backup paths, restoring partial connectivity."""
        countered = self._get_countered_attacks(env, action["counters"])
        if not countered:
            return False, []

        # Simulate: restore status on downed interfaces, reduce packet loss
        for i in range(N_INTERFACES):
            if env._status[i] < 0.5:
                env._status[i] = 0.7  # Partial restoration via alternate path
            env._pkt_loss[i] = max(0.0, env._pkt_loss[i] - 0.4)

        # Restore GNS3 links for affected attacks
        affected = self._affected_ifaces(env, action["counters"])
        self._restore_ifaces(env, affected)

        cleared = self._clear_attacks(env, action["counters"])
        self._decay_anomaly(env, 0.3)
        return True, cleared

    def _failover_activation(self, action, env):
        """Activate redundant core links when primary core is under attack."""
        countered = self._get_countered_attacks(env, action["counters"])
        if not countered:
            return False, []

        # Restore core interfaces
        for (node, port) in CORE_INTERFACES:
            env.gns3.restore_interface(node, port)
            env.interfaces_down.discard((node, port))

        # Restore core interface status in synthetic state
        for i, (node, port) in enumerate(INTERFACE_ACTIONS):
            if (node, port) in CORE_INTERFACES and i < N_INTERFACES:
                env._status[i] = 1.0

        # Reduce DDoS / shutdown impact
        for i in range(N_INTERFACES):
            env._pkt_loss[i] = max(0.0, env._pkt_loss[i] - 0.6)
            env._traffic[i]  = max(0.0, env._traffic[i] - 2.0)

        cleared = self._clear_attacks(env, action["counters"])
        self._decay_anomaly(env, 0.4)
        return True, cleared

    def _config_restoration(self, action, env):
        """Restore original device configuration after tampering / backdoor."""
        countered = self._get_countered_attacks(env, action["counters"])
        if not countered and not env._config_tampered and not env._backdoor_active:
            return False, []

        env._config_tampered  = False
        env._backdoor_active  = False
        env._firewall_bypassed = False

        # Reduce errors and packet loss from tampered configs
        env._errors   = np.clip(env._errors   - 2.0, 0, 10)
        env._pkt_loss = np.clip(env._pkt_loss  - 0.4, 0, 1)

        affected = self._affected_ifaces(env, action["counters"])
        self._restore_ifaces(env, affected)
        cleared = self._clear_attacks(env, action["counters"])
        self._decay_anomaly(env, 0.35)
        return True, cleared

    def _full_network_lockdown(self, action, env):
        """Emergency: isolate entire network segment, wipe active persistence."""
        countered = self._get_countered_attacks(env, action["counters"])
        if not countered:
            return False, []

        env._config_tampered   = False
        env._backdoor_active   = False
        env._firewall_bypassed = False

        # Heavy traffic/error reduction — lockdown drops everything
        env._traffic          = np.clip(env._traffic - 3.0, 0, None)
        env._errors           = np.clip(env._errors  - 3.0, 0, 10)
        env._pkt_loss         = np.clip(env._pkt_loss - 0.5, 0, 1)
        env._broadcast_ratio  = np.clip(env._broadcast_ratio - 0.5, 0, 1)

        affected = self._affected_ifaces(env, action["counters"])
        self._restore_ifaces(env, affected)
        cleared = self._clear_attacks(env, action["counters"])
        self._decay_anomaly(env, 0.6)
        return True, cleared

    # ── Immediate response actions ───────────────────────

    def _port_isolation(self, action, env):
        """Quarantine an infected port — stops MAC flood / VLAN hopping spread."""
        countered = self._get_countered_attacks(env, action["counters"])
        if not countered:
            return False, []

        # Isolating port: reduce broadcast storm and traffic on affected interfaces
        affected = self._affected_ifaces(env, action["counters"])
        for i in affected:
            env._broadcast_ratio[i] = max(0.0, env._broadcast_ratio[i] - 0.6)
            env._traffic[i]         = max(0.0, env._traffic[i] - 2.0)
            env._errors[i]          = max(0.0, env._errors[i] - 1.5)

        self._restore_ifaces(env, affected)
        cleared = self._clear_attacks(env, action["counters"])
        self._decay_anomaly(env, 0.35, affected)
        return True, cleared

    def _mac_blacklisting(self, action, env):
        """Block specific MAC addresses causing the flood or poisoning."""
        countered = self._get_countered_attacks(env, action["counters"])
        if not countered:
            return False, []

        affected = self._affected_ifaces(env, action["counters"])
        for i in affected:
            env._broadcast_ratio[i] = max(0.0, env._broadcast_ratio[i] - 0.5)
            env._arp_size[i]        = max(0.0, env._arp_size[i] - 0.4)
            env._errors[i]          = max(0.0, env._errors[i] - 1.0)

        self._restore_ifaces(env, affected)
        cleared = self._clear_attacks(env, action["counters"])
        self._decay_anomaly(env, 0.3, affected)
        return True, cleared

    def _rate_limiting(self, action, env):
        """Throttle traffic on links under DDoS attack."""
        countered = self._get_countered_attacks(env, action["counters"])
        if not countered:
            return False, []

        affected = self._affected_ifaces(env, action["counters"])
        for i in affected:
            env._traffic[i]   = max(0.0, env._traffic[i] - 3.5)
            env._pkt_loss[i]  = max(0.0, env._pkt_loss[i] - 0.4)
            env._cpu[i]       = max(0.0, env._cpu[i] - 0.3)

        self._restore_ifaces(env, affected)
        cleared = self._clear_attacks(env, action["counters"])
        self._decay_anomaly(env, 0.4, affected)
        return True, cleared

    def _icmp_blocking(self, action, env):
        """Drop ICMP flood traffic to reduce DDoS impact."""
        countered = self._get_countered_attacks(env, action["counters"])
        if not countered:
            return False, []

        affected = self._affected_ifaces(env, action["counters"])
        for i in affected:
            env._traffic[i]  = max(0.0, env._traffic[i] - 2.0)
            env._errors[i]   = max(0.0, env._errors[i]  - 1.5)
            env._pkt_loss[i] = max(0.0, env._pkt_loss[i] - 0.3)

        self._restore_ifaces(env, affected)
        cleared = self._clear_attacks(env, action["counters"])
        self._decay_anomaly(env, 0.3, affected)
        return True, cleared

    def _stp_guard_activation(self, action, env):
        """Enable BPDU Guard and Root Guard to stop STP manipulation."""
        countered = self._get_countered_attacks(env, action["counters"])
        if not countered:
            return False, []

        # STP attack affects all ifaces — restore all
        for i in range(N_INTERFACES):
            env._broadcast_ratio[i] = max(0.0, env._broadcast_ratio[i] - 0.55)
            env._traffic[i]         = max(0.0, env._traffic[i] - 2.0)
            env._errors[i]          = max(0.0, env._errors[i] - 1.5)

        self._restore_ifaces(env, list(range(N_INTERFACES)))
        cleared = self._clear_attacks(env, action["counters"])
        self._decay_anomaly(env, 0.5)
        return True, cleared

    def _arp_inspection_enable(self, action, env):
        """Enable Dynamic ARP Inspection to validate ARP packets."""
        countered = self._get_countered_attacks(env, action["counters"])
        if not countered:
            return False, []

        affected = self._affected_ifaces(env, action["counters"])
        for i in affected:
            env._arp_size[i]        = max(0.0, env._arp_size[i] - 0.55)
            env._broadcast_ratio[i] = max(0.0, env._broadcast_ratio[i] - 0.3)
            env._pkt_loss[i]        = max(0.0, env._pkt_loss[i] - 0.2)

        self._restore_ifaces(env, affected)
        cleared = self._clear_attacks(env, action["counters"])
        self._decay_anomaly(env, 0.35, affected)
        return True, cleared

    def _firewall_rule_update(self, action, env):
        """Push new ACL rules to block Firewall Bypass / VLAN Pivot traffic."""
        countered = self._get_countered_attacks(env, action["counters"])
        if not countered:
            return False, []

        env._firewall_bypassed = False
        affected = self._affected_ifaces(env, action["counters"])
        for i in affected:
            env._traffic[i]  = max(0.0, env._traffic[i] - 1.5)
            env._pkt_loss[i] = max(0.0, env._pkt_loss[i] - 0.2)

        self._restore_ifaces(env, affected)
        cleared = self._clear_attacks(env, action["counters"])
        self._decay_anomaly(env, 0.3, affected)
        return True, cleared

    # ── Proactive / Intelligent actions ──────────────────

    def _honeypot_deployment(self, action, env):
        """
        Deploy a honeypot to trap reconnaissance attacks.
        If Red is currently in a recon phase, the honeypot is triggered
        immediately and Red suffers the HONEYPOT_PENALTY.
        """
        env.honeypot_active = True

        countered = self._get_countered_attacks(env, action["counters"])
        if countered:
            # Red walked into the honeypot
            env.honeypot_triggered = True
            affected = self._affected_ifaces(env, action["counters"])
            self._restore_ifaces(env, affected)
            cleared = self._clear_attacks(env, action["counters"])
            self._decay_anomaly(env, 0.2)
            return True, cleared

        # No recon currently active — honeypot is pre-deployed for future use
        return True, []

    def _anomaly_detection_alert(self, action, env):
        """
        Trigger anomaly detection — raises awareness of current anomaly level.
        Counts as a proactive detection if anomaly score is already elevated.
        """
        countered = self._get_countered_attacks(env, action["counters"])
        avg_anomaly = float(np.mean(env._anomaly))

        if countered or avg_anomaly > 0.3:
            # Mark that Blue detected the attack proactively
            env.attack_detected_step = env._step_count
            affected = self._affected_ifaces(env, action["counters"])
            self._restore_ifaces(env, affected)
            cleared = self._clear_attacks(env, action["counters"])
            self._decay_anomaly(env, 0.25)
            return True, cleared

        return False, []

    def _dynamic_acl(self, action, env):
        """Automatically create ACL rules based on detected attack pattern."""
        countered = self._get_countered_attacks(env, action["counters"])
        if not countered:
            return False, []

        affected = self._affected_ifaces(env, action["counters"])
        for i in affected:
            env._broadcast_ratio[i] = max(0.0, env._broadcast_ratio[i] - 0.35)
            env._traffic[i]         = max(0.0, env._traffic[i] - 1.0)
            env._arp_size[i]        = max(0.0, env._arp_size[i] - 0.3)

        # Dynamic ACL also prevents firewall bypass going forward
        env._firewall_bypassed = False

        self._restore_ifaces(env, affected)
        cleared = self._clear_attacks(env, action["counters"])
        self._decay_anomaly(env, 0.3, affected)
        return True, cleared

    def _traffic_baselining(self, action, env):
        """Compare current traffic to baseline, flag deviations."""
        countered = self._get_countered_attacks(env, action["counters"])
        avg_anomaly = float(np.mean(env._anomaly))

        if countered or avg_anomaly > 0.25:
            env.attack_detected_step = env._step_count
            # Reduce anomaly slightly — baselining helps identify normal vs abnormal
            self._decay_anomaly(env, 0.2)
            affected = self._affected_ifaces(env, action["counters"])
            self._restore_ifaces(env, affected)
            cleared = self._clear_attacks(env, action["counters"])
            return True, cleared

        return False, []

    def _ids_alert_response(self, action, env):
        """Respond to IDS alert — investigates and counters recon / lateral movement."""
        countered = self._get_countered_attacks(env, action["counters"])
        if not countered:
            return False, []

        if env.attack_detected_step is None:
            env.attack_detected_step = env._step_count

        affected = self._affected_ifaces(env, action["counters"])
        for i in affected:
            env._traffic[i]  = max(0.0, env._traffic[i] - 0.8)
            env._cpu[i]      = max(0.0, env._cpu[i] - 0.2)

        self._restore_ifaces(env, affected)
        cleared = self._clear_attacks(env, action["counters"])
        self._decay_anomaly(env, 0.25, affected)
        return True, cleared

    def _vlan_reassignment(self, action, env):
        """Move devices to clean VLANs after VLAN hop/pivot attack."""
        countered = self._get_countered_attacks(env, action["counters"])
        if not countered:
            return False, []

        affected = self._affected_ifaces(env, action["counters"])
        for i in affected:
            env._broadcast_ratio[i] = max(0.0, env._broadcast_ratio[i] - 0.45)
            env._arp_size[i]        = max(0.0, env._arp_size[i] - 0.4)
            env._pkt_loss[i]        = max(0.0, env._pkt_loss[i] - 0.2)

        self._restore_ifaces(env, affected)
        cleared = self._clear_attacks(env, action["counters"])
        self._decay_anomaly(env, 0.3, affected)
        return True, cleared

    # ── Helpers ──────────────────────────────────────────

    def _get_countered_attacks(self, env, counters: list) -> list:
        """Return list of currently-active attack names that this action counters."""
        return [atk for atk in env.active_attacks if atk in counters]

    def _affected_ifaces(self, env, counters: list) -> list:
        """Return union of affected interface indices across all countered attacks."""
        ifaces = set()
        for atk_name, atk_info in env.active_attacks.items():
            if atk_name in counters:
                ifaces.update(atk_info.get("affected", []))
        return list(ifaces) if ifaces else list(range(N_INTERFACES))

    def _clear_attacks(self, env, counters: list) -> list:
        """Remove countered attacks from env.active_attacks. Return cleared names."""
        cleared = []
        for name in list(env.active_attacks.keys()):
            if name in counters:
                del env.active_attacks[name]
                cleared.append(name)
        return cleared

    def _decay_anomaly(self, env, amount: float, ifaces: list = None):
        """Reduce anomaly scores after a successful defense."""
        if ifaces:
            for i in ifaces:
                env._anomaly[i] = max(0.0, env._anomaly[i] - amount)
        else:
            env._anomaly[:] = np.clip(env._anomaly - amount, 0, 1)

    def _restore_ifaces(self, env, ifaces: list):
        """Restore real GNS3 links for interface indices."""
        for idx in ifaces:
            if idx >= len(INTERFACE_ACTIONS):
                continue
            node, port = INTERFACE_ACTIONS[idx]
            if (node, port) in env.interfaces_down:
                env.gns3.restore_interface(node, port)
                env.interfaces_down.discard((node, port))
            if idx < N_INTERFACES:
                env._status[idx] = 1.0

    def _default(self, action, env):
        """Fallback for unmapped actions."""
        return False, []
