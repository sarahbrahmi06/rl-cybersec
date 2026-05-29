"""
Attack Executor
===============
Simulates the effect of every Red-agent attack action on the environment.

Design
------
- Each method corresponds to one entry in RED_ACTIONS from config.py.
- Returns a tuple: (success: bool, affected_ifaces: list[int], severity: float)
- Link-level attacks (Shutdown Link / Shutdown Core Link) make real GNS3 API calls.
- All other attacks are simulated numerically:
    * They perturb the environment's internal synthetic-state arrays.
    * In "influx" mode the same perturbations happen — the InfluxDB telemetry
      will not change (we don't control the real traffic), but the reward system
      and episode metrics still reflect the attack semantics correctly.
"""

import random
import numpy as np

from config import (
    INTERFACE_ACTIONS, CORE_INTERFACES,
    N_INTERFACES, RED_ACTIONS,
    ACCESS_IFACE_IDX, SWITCH_IFACE_IDX, ROUTER_IFACE_IDX,
    FIREWALL_IFACE_IDX, DEVICE_IFACE_IDX, NETWORK_IFACE_IDX,
    CORE_IFACE_IDX, VICTIM_IFACE_IDX, MGMT_IFACE_IDX,
)


class AttackExecutor:
    """Executes Red-agent attacks and updates environment internal state."""

    def __init__(self, gns3_connector):
        self.gns3 = gns3_connector

        # Dispatch table: action name → handler method
        self._dispatch = {
            "Shutdown Link":      self._shutdown_link,
            "MAC Flooding":       self._mac_flooding,
            "ARP Poisoning":      self._arp_poisoning,
            "DDoS Simulation":    self._ddos_simulation,
            "CPU Exhaustion":     self._cpu_exhaustion,
            "VLAN Hopping":       self._vlan_hopping,
            "STP Attack":         self._stp_attack,
            "Port Scan":          self._port_scan,
            "Topology Mapping":   self._topology_mapping,
            "Traffic Sniffing":   self._traffic_sniffing,
            "Backdoor Creation":  self._backdoor_creation,
            "Config Tampering":   self._config_tampering,
            "Shutdown Core Link": self._shutdown_core_link,
            "DDoS Core Router":   self._ddos_core_router,
            "Credential Theft":   self._credential_theft,
            "Lateral Movement":   self._lateral_movement,
            "Firewall Bypass":    self._firewall_bypass,
            "VLAN Pivot":         self._vlan_pivot,
        }

    # ── Public API ───────────────────────────────────────

    def execute(self, action_dict: dict, env) -> tuple:
        """
        Execute the given attack action against the environment.

        Parameters
        ----------
        action_dict : dict
            An entry from RED_ACTIONS.
        env : NetworkEnvironment
            The live environment (state is read and mutated here).

        Returns
        -------
        (success, affected_ifaces, severity)
        """
        name    = action_dict["name"]
        handler = self._dispatch.get(name, self._default)
        return handler(action_dict, env)

    # ── Link-level attacks (real GNS3 calls) ────────────

    def _shutdown_link(self, action, env):
        """Shut down a randomly selected non-core interface."""
        candidates = [
            i for i, (node, port) in enumerate(INTERFACE_ACTIONS)
            if (node, port) not in CORE_INTERFACES
            and (node, port) not in env.interfaces_down
        ]
        if not candidates:
            return False, [], 0.0

        idx  = random.choice(candidates)
        affected = self._shutdown_ifaces(env, [idx])
        if not affected:
            return False, [], 0.0
        self._register_attack(env, action, affected)
        return True, affected, action["severity"]

    def _shutdown_core_link(self, action, env):
        """Shut down a core uplink — highest value target."""
        candidates = [
            i for i, (node, port) in enumerate(INTERFACE_ACTIONS)
            if (node, port) in CORE_INTERFACES
            and (node, port) not in env.interfaces_down
        ]
        if not candidates:
            return False, [], 0.0

        idx  = random.choice(candidates)
        affected = self._shutdown_ifaces(env, [idx])
        if not affected:
            return False, [], 0.0
        self._register_attack(env, action, affected)
        return True, affected, action["severity"]

    # ── Synthetic network attacks ────────────────────────

    def _mac_flooding(self, action, env):
        """
        MAC flooding — overwhelms CAM table, forces switch to broadcast all
        frames, causing high traffic and broadcast storm.
        """
        targets = self._pick_ifaces(env, n=2, pool=ACCESS_IFACE_IDX)
        affected = self._shutdown_ifaces(env, targets)
        if not affected:
            return False, [], 0.0
        sev     = action["severity"]

        # High broadcast ratio + elevated traffic on affected interfaces
        for i in affected:
            env._broadcast_ratio[i] = min(1.0, env._broadcast_ratio[i] + sev * 0.8)
            env._traffic[i]         = np.log1p(env._traffic[i] * (1 + sev * 5))
            env._errors[i]          = min(10.0, env._errors[i] + sev * 3)

        # Raise anomaly — very noisy attack
        env._anomaly[affected] = np.clip(env._anomaly[affected] + 0.6, 0, 1)
        self._register_attack(env, action, affected)
        return True, affected, sev

    def _arp_poisoning(self, action, env):
        """
        ARP poisoning — grows ARP table abnormally, redirects traffic
        (man-in-the-middle), subtle anomaly.
        """
        targets = self._pick_ifaces(env, n=2, pool=ACCESS_IFACE_IDX)
        affected = self._shutdown_ifaces(env, targets)
        if not affected:
            return False, [], 0.0
        sev     = action["severity"]

        for i in affected:
            env._arp_size[i]   = min(1.0, env._arp_size[i] + sev * 0.7)
            env._traffic[i]   *= (1 + sev * 0.3)  # slight traffic increase (MITM copy)
            env._pkt_loss[i]   = min(1.0, env._pkt_loss[i] + sev * 0.2)

        # Subtle anomaly — high stealth attack
        env._anomaly[affected] = np.clip(env._anomaly[affected] + 0.35, 0, 1)
        self._register_attack(env, action, affected)
        return True, affected, sev

    def _ddos_simulation(self, action, env):
        """
        DDoS — saturates links with massive traffic, causes packet loss
        and high error counts.
        """
        targets = self._pick_ifaces(env, n=1, pool=ROUTER_IFACE_IDX)
        affected = self._shutdown_ifaces(env, targets)
        if not affected:
            return False, [], 0.0
        sev     = action["severity"]

        for i in affected:
            env._traffic[i]   = np.log1p(1e6 * sev)    # saturate bandwidth
            env._pkt_loss[i]  = min(1.0, env._pkt_loss[i] + sev * 0.6)
            env._errors[i]    = min(10.0, env._errors[i] + sev * 4)
            env._cpu[i]       = min(1.0, env._cpu[i] + sev * 0.5)

        # Very noisy — low stealth
        env._anomaly[affected] = np.clip(env._anomaly[affected] + 0.75, 0, 1)
        self._register_attack(env, action, affected)
        return True, affected, sev

    def _cpu_exhaustion(self, action, env):
        """
        CPU exhaustion — sends complex packets to max out device CPU,
        degrades forwarding performance.
        """
        targets = self._pick_ifaces(env, n=1, pool=DEVICE_IFACE_IDX)
        affected = self._shutdown_ifaces(env, targets)
        if not affected:
            return False, [], 0.0
        sev     = action["severity"]

        for i in affected:
            env._cpu[i]      = min(1.0, env._cpu[i] + sev * 0.85)
            env._pkt_loss[i] = min(1.0, env._pkt_loss[i] + sev * 0.35)
            env._traffic[i] *= (1 + sev * 0.1)

        env._anomaly[affected] = np.clip(env._anomaly[affected] + 0.5, 0, 1)
        self._register_attack(env, action, affected)
        return True, affected, sev

    def _vlan_hopping(self, action, env):
        """
        VLAN hopping — double-tagging to jump to a different VLAN,
        breaks segmentation, moderate stealth.
        """
        targets = self._pick_ifaces(env, n=1, pool=ACCESS_IFACE_IDX)
        affected = self._shutdown_ifaces(env, targets)
        if not affected:
            return False, [], 0.0
        sev     = action["severity"]

        for i in affected:
            env._broadcast_ratio[i] = min(1.0, env._broadcast_ratio[i] + sev * 0.45)
            env._traffic[i]        *= (1 + sev * 0.4)
            env._arp_size[i]        = min(1.0, env._arp_size[i] + sev * 0.3)

        env._anomaly[affected] = np.clip(env._anomaly[affected] + 0.4, 0, 1)
        self._register_attack(env, action, affected)
        return True, affected, sev

    def _stp_attack(self, action, env):
        """
        STP attack — sends crafted BPDUs to become root bridge, rerouting
        all traffic through attacker. Affects all interfaces.
        """
        targets = list(ACCESS_IFACE_IDX)
        affected = self._shutdown_ifaces(env, targets)
        if not affected:
            return False, [], 0.0
        sev     = action["severity"]

        for i in affected:
            env._broadcast_ratio[i] = min(1.0, env._broadcast_ratio[i] + sev * 0.55)
            env._traffic[i]        *= (1 + sev * 0.6)
            env._errors[i]          = min(10.0, env._errors[i] + sev * 2)

        env._anomaly[affected] = np.clip(env._anomaly[affected] + 0.55, 0, 1)
        self._register_attack(env, action, affected)
        return True, affected, sev

    # ── Reconnaissance attacks ───────────────────────────

    def _port_scan(self, action, env):
        """
        Port scanning — probes open ports. Slight traffic increase,
        very high stealth. Advances kill chain Stage 1.
        """
        targets = self._pick_ifaces(env, n=1, pool=VICTIM_IFACE_IDX)
        affected = self._shutdown_ifaces(env, targets)
        if not affected:
            return False, [], 0.0
        sev     = action["severity"]

        for i in affected:
            env._traffic[i] = np.log1p(env._traffic[i] * (1 + sev * 0.15))

        # Very low anomaly — high stealth
        env._anomaly[affected] = np.clip(env._anomaly[affected] + 0.15, 0, 1)
        self._register_attack(env, action, affected)
        return True, affected, sev

    def _topology_mapping(self, action, env):
        """
        Topology mapping — ICMP/SNMP sweep to learn network layout.
        Minimal footprint. Advances kill chain Stage 1.
        """
        targets = self._pick_ifaces(env, n=1, pool=MGMT_IFACE_IDX)
        affected = self._shutdown_ifaces(env, targets)
        if not affected:
            return False, [], 0.0
        sev     = action["severity"]

        # Tiny broadcast increase (ICMP sweep)
        env._broadcast_ratio[affected] = np.clip(env._broadcast_ratio[affected] + sev * 0.1, 0, 1)
        env._anomaly[affected] = np.clip(env._anomaly[affected] + 0.1, 0, 1)
        self._register_attack(env, action, affected)
        return True, affected, sev

    def _traffic_sniffing(self, action, env):
        """
        Traffic sniffing — passive monitoring of packets.
        Near-zero footprint but updates internal credential state.
        Advances kill chain Stage 1.
        """
        targets = self._pick_ifaces(env, n=1, pool=VICTIM_IFACE_IDX)
        affected = self._shutdown_ifaces(env, targets)
        if not affected:
            return False, [], 0.0
        sev     = action["severity"]

        # Almost invisible
        env._anomaly[affected] = np.clip(env._anomaly[affected] + 0.08, 0, 1)
        self._register_attack(env, action, affected)
        return True, affected, sev

    def _credential_theft(self, action, env):
        """
        Credential theft via captured packets / protocol exploitation.
        Advances kill chain Stage 1.
        """
        targets = self._pick_ifaces(env, n=1, pool=VICTIM_IFACE_IDX)
        affected = self._shutdown_ifaces(env, targets)
        if not affected:
            return False, [], 0.0
        sev     = action["severity"]

        env._arp_size[affected] = np.clip(
            [env._arp_size[i] + sev * 0.25 for i in affected], 0, 1
        )
        env._anomaly[affected] = np.clip(env._anomaly[affected] + 0.2, 0, 1)
        self._register_attack(env, action, affected)
        return True, affected, sev

    # ── Persistence attacks ──────────────────────────────

    def _backdoor_creation(self, action, env):
        """
        Backdoor — installs persistent access, survives link restore.
        Persists as a flagged 'backdoor_active' state in env.
        """
        targets = self._pick_ifaces(env, n=1, pool=VICTIM_IFACE_IDX)
        affected = self._shutdown_ifaces(env, targets)
        if not affected:
            return False, [], 0.0
        sev     = action["severity"]

        env._backdoor_active = True
        env._cpu[affected]    = np.clip([env._cpu[i] + sev * 0.25 for i in affected], 0, 1)
        env._anomaly[affected] = np.clip(env._anomaly[affected] + 0.3, 0, 1)
        self._register_attack(env, action, affected)
        return True, affected, sev

    def _config_tampering(self, action, env):
        """
        Configuration tampering — modifies ACLs / routing tables so
        device misbehaves even after Blue restores links.
        Persists as 'config_tampered' in env.
        """
        targets = self._pick_ifaces(env, n=1, pool=CORE_IFACE_IDX)
        affected = self._shutdown_ifaces(env, targets)
        if not affected:
            return False, [], 0.0
        sev     = action["severity"]

        env._config_tampered = True
        for i in affected:
            env._pkt_loss[i] = min(1.0, env._pkt_loss[i] + sev * 0.45)
            env._errors[i]   = min(10.0, env._errors[i] + sev * 2)

        env._anomaly[affected] = np.clip(env._anomaly[affected] + 0.35, 0, 1)
        self._register_attack(env, action, affected)
        return True, affected, sev

    def _lateral_movement(self, action, env):
        """
        Lateral movement — pivots from compromised device to adjacent ones.
        Advances kill chain Stage 3.
        """
        targets = self._pick_ifaces(env, n=min(3, len(ACCESS_IFACE_IDX) + len(VICTIM_IFACE_IDX)),
                                    pool=ACCESS_IFACE_IDX + VICTIM_IFACE_IDX)
        affected = self._shutdown_ifaces(env, targets)
        if not affected:
            return False, [], 0.0
        sev     = action["severity"]

        for i in affected:
            env._cpu[i]   = min(1.0, env._cpu[i] + sev * 0.3)
            env._pkt_loss[i] = min(1.0, env._pkt_loss[i] + sev * 0.2)

        env._anomaly[affected] = np.clip(env._anomaly[affected] + 0.3, 0, 1)
        self._register_attack(env, action, affected)
        return True, affected, sev

    def _firewall_bypass(self, action, env):
        """
        Firewall bypass — exploits misconfiguration or tunnelling to
        pass through the (simulated) firewall node.
        """
        targets = self._pick_ifaces(env, n=1, pool=FIREWALL_IFACE_IDX)
        affected = self._shutdown_ifaces(env, targets)
        if not affected:
            return False, [], 0.0
        sev     = action["severity"]

        env._firewall_bypassed = True
        for i in affected:
            env._traffic[i] *= (1 + sev * 0.3)

        env._anomaly[affected] = np.clip(env._anomaly[affected] + 0.4, 0, 1)
        self._register_attack(env, action, affected)
        return True, affected, sev

    def _vlan_pivot(self, action, env):
        """
        VLAN pivot — after hopping VLANs, now fully controls inter-VLAN
        routing. High severity, moderate stealth.
        """
        targets = self._pick_ifaces(env, n=2, pool=ACCESS_IFACE_IDX)
        affected = self._shutdown_ifaces(env, targets)
        if not affected:
            return False, [], 0.0
        sev     = action["severity"]

        for i in affected:
            env._broadcast_ratio[i] = min(1.0, env._broadcast_ratio[i] + sev * 0.5)
            env._arp_size[i]        = min(1.0, env._arp_size[i] + sev * 0.45)

        env._anomaly[affected] = np.clip(env._anomaly[affected] + 0.45, 0, 1)
        self._register_attack(env, action, affected)
        return True, affected, sev

    def _ddos_core_router(self, action, env):
        """
        DDoS the core router — overwhelms core infrastructure.
        Highest severity, very low stealth. Advances kill chain Stage 4.
        """
        # Targets all interfaces (core router affects everything)
        targets = list(CORE_IFACE_IDX)
        affected = self._shutdown_ifaces(env, targets)
        if not affected:
            return False, [], 0.0
        sev     = action["severity"]

        for i in affected:
            env._traffic[i]    = np.log1p(5e6 * sev)
            env._pkt_loss[i]   = min(1.0, env._pkt_loss[i] + sev * 0.8)
            env._cpu[i]        = min(1.0, env._cpu[i] + sev * 0.7)
            env._errors[i]     = min(10.0, env._errors[i] + sev * 5)
            env._status[i]     = 0.3  # near-down but not fully down

        env._anomaly[affected] = np.clip(env._anomaly[affected] + 0.9, 0, 1)

        self._register_attack(env, action, affected)
        return True, affected, sev

    # ── Helpers ──────────────────────────────────────────

    def _pick_ifaces(self, env, n: int, pool: list) -> list:
        """Pick n interface indices from a pool, preferring currently-up links."""
        if not pool:
            return []
        up = [i for i in pool if env._status[i] > 0.5]
        candidates = up if up else list(pool)
        return random.sample(candidates, min(n, len(candidates)))

    def _shutdown_ifaces(self, env, ifaces: list) -> list:
        """Shutdown real GNS3 links for given interface indices."""
        affected = []
        for idx in ifaces:
            if idx >= len(INTERFACE_ACTIONS):
                continue
            node, port = INTERFACE_ACTIONS[idx]
            if (node, port) in env.interfaces_down:
                continue
            success = env.gns3.shutdown_interface(node, port)
            if success:
                env.interfaces_down.add((node, port))
                if idx < N_INTERFACES:
                    env._status[idx] = 0.0
                affected.append(idx)
        return affected

    def _register_attack(self, env, action: dict, affected: list):
        """Log the active attack in the environment's tracking dict."""
        env.active_attacks[action["name"]] = {
            "action":   action,
            "affected": affected,
            "step":     env._step_count,
            "severity": action["severity"],
        }

    def _default(self, action, env):
        """Fallback for unmapped actions."""
        return False, [], 0.0
