"""
Centralized Configuration
=========================
All connection strings, credentials, and hyperparameters in one place.
Edit this file to match your environment — no other file needs changing.
"""

# ─────────────────────────────────────────────
# GNS3 Server
# ─────────────────────────────────────────────
GNS3_BASE_URL   = "http://10.202.91.125:3080/v3"
GNS3_PROJECT_ID = "1e81b342-9ba3-40a4-b48c-12fdd94787bc"
GNS3_TOKEN      = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9"
    ".eyJzdWIiOiJhZG1pbiIsImV4cCI6MTc3NjAyMTI5Nn0"
    ".As8kAqdssmTTZ3Z010Tgg6MBvntRM2NQtcKpSUFALM8"
)

# ─────────────────────────────────────────────
# InfluxDB (Telemetry)
# ─────────────────────────────────────────────
INFLUXDB_URL    = "http://10.202.91.55:8086"
INFLUXDB_TOKEN  = "project2026token"
INFLUXDB_ORG    = "project"
INFLUXDB_BUCKET = "network"

# ─────────────────────────────────────────────
# Telemetry Mode
# ─────────────────────────────────────────────
# "influx"    → live telemetry from InfluxDB (production)
# "synthetic" → generated telemetry for dry-run training (no InfluxDB required)
TELEMETRY_MODE  = "influx"
SYNTHETIC_SEED  = 42

# ─────────────────────────────────────────────
# Network Topology — Interface-level Actions
# ─────────────────────────────────────────────
# Used for Shutdown Link / Restore Link actions (real GNS3 calls).
# Each tuple: (node_name_in_GNS3, adapter_number)
#
# Mapped from live GNS3 links:
# - Switch1 Gi0/3  ↔ SwitchL31 Gi0/0
# - Switch2 Gi0/2  ↔ SwitchL31 Gi0/1
# - Switch3 Gi0/2  ↔ SwitchL32 Gi0/0
# - Switch4 Gi0/1  ↔ SwitchL32 Gi0/1
# - Router1 Gi0/0  ↔ SwitchL31 Gi0/2
# - Router2 Gi0/0  ↔ SwitchL32 Gi0/2
# - FW eth0        ↔ Router1 Gi0/1
# - FW eth1        ↔ Router2 Gi0/1
# - SwitchL32 Gi0/3 ↔ Victim-1 eth0
# - SwitchL31 Gi0/3 ↔ Cloud1 tap0 (mgmt)
INTERFACE_ACTIONS = [
    ("Switch1", 3),              # Trunk up (VLAN10/20)
    ("Switch2", 2),              # Trunk up (VLAN10/20)
    ("Switch3", 2),              # Trunk up (VLAN30)
    ("Switch4", 1),              # Trunk up (VLAN50)
    ("Router1", 0),              # Core downlink
    ("Router2", 0),              # Core downlink
    ("Alpine-Firewall-1", 0),    # FW ↔ Router1
    ("Alpine-Firewall-1", 1),    # FW ↔ Router2
    ("SwitchL32", 3),            # Victim-1 segment
    ("SwitchL31", 3),            # Mgmt / tap segment
]

# Core interfaces — highest value targets for Red, highest priority for Blue
CORE_INTERFACES = [("Router1", 0), ("Router2", 0)]

# Future GNS3 topology nodes (not yet deployed — simulated for now)
FIREWALL_NODE = "Alpine-Firewall-1"
IDS_NODE      = "IDS1"
DMZ_NODE      = "DMZ1"

# ─────────────────────────────────────────────
# Interfaces Monitored via SNMP / InfluxDB
# ─────────────────────────────────────────────
# These names must match the "name" tag in InfluxDB.
# If your Influx tags use a different format, update this list accordingly.
MONITORED_INTERFACES = [
    "GigabitEthernet0/0",
    "GigabitEthernet0/1",
    "GigabitEthernet0/2",
    "GigabitEthernet0/3",
    "GigabitEthernet1/0",
    "GigabitEthernet1/1",
    "GigabitEthernet1/2",
    "GigabitEthernet1/3",
    "Vlan10",
    "Vlan20",
]
N_INTERFACES = len(MONITORED_INTERFACES)   # 10

# Interface index groups (aligned with INTERFACE_ACTIONS order)
ACCESS_IFACE_IDX   = [0, 1, 2, 3]          # Access/distribution trunks
CORE_IFACE_IDX     = [4, 5]                # Router core links
FIREWALL_IFACE_IDX = [6, 7]                # Perimeter links
VICTIM_IFACE_IDX   = [8]                   # Victim segment
MGMT_IFACE_IDX     = [9]                   # Mgmt/tap segment
SWITCH_IFACE_IDX   = ACCESS_IFACE_IDX + VICTIM_IFACE_IDX + MGMT_IFACE_IDX
ROUTER_IFACE_IDX   = CORE_IFACE_IDX
DEVICE_IFACE_IDX   = CORE_IFACE_IDX + FIREWALL_IFACE_IDX
NETWORK_IFACE_IDX  = list(range(N_INTERFACES))

# ─────────────────────────────────────────────
# Red Agent — Attack Action Catalog (18 total)
# ─────────────────────────────────────────────
# Fields:
#   id             : action index (0–17)
#   name           : human-readable label (used in logs and reward routing)
#   category       : "disruption" | "recon" | "persistence"
#   target         : "interface" | "switch" | "router" | "core" | "device" | "network" | "firewall"
#   base_reward    : base reward on success
#   severity       : 0–1, how damaging the attack is (affects state perturbation)
#   stealth        : 0–1, how hard it is for Blue to detect (affects anomaly score)
#   core_multiplier: reward multiplier when hitting core infrastructure
RED_ACTIONS = [
    # ── Network Disruption ──────────────────────────────────────────────────
    {"id":  0, "name": "Shutdown Link",      "category": "disruption",  "target": "interface",
     "base_reward": 10.0, "severity": 0.60, "stealth": 0.70, "core_multiplier": 2.0},
    {"id":  1, "name": "MAC Flooding",       "category": "disruption",  "target": "switch",
     "base_reward": 12.0, "severity": 0.70, "stealth": 0.40, "core_multiplier": 1.5},
    {"id":  2, "name": "ARP Poisoning",      "category": "disruption",  "target": "switch",
     "base_reward": 15.0, "severity": 0.80, "stealth": 0.60, "core_multiplier": 1.5},
    {"id":  3, "name": "DDoS Simulation",    "category": "disruption",  "target": "router",
     "base_reward": 12.0, "severity": 0.75, "stealth": 0.30, "core_multiplier": 2.0},
    {"id":  4, "name": "CPU Exhaustion",     "category": "disruption",  "target": "device",
     "base_reward": 11.0, "severity": 0.65, "stealth": 0.50, "core_multiplier": 1.5},
    {"id":  5, "name": "VLAN Hopping",       "category": "disruption",  "target": "switch",
     "base_reward": 14.0, "severity": 0.75, "stealth": 0.70, "core_multiplier": 1.5},
    {"id":  6, "name": "STP Attack",         "category": "disruption",  "target": "switch",
     "base_reward": 16.0, "severity": 0.85, "stealth": 0.60, "core_multiplier": 2.0},
    # ── Reconnaissance ─────────────────────────────────────────────────────
    {"id":  7, "name": "Port Scan",          "category": "recon",       "target": "network",
     "base_reward":  5.0, "severity": 0.20, "stealth": 0.80, "core_multiplier": 1.0},
    {"id":  8, "name": "Topology Mapping",   "category": "recon",       "target": "network",
     "base_reward":  6.0, "severity": 0.25, "stealth": 0.75, "core_multiplier": 1.0},
    {"id":  9, "name": "Traffic Sniffing",   "category": "recon",       "target": "interface",
     "base_reward":  7.0, "severity": 0.30, "stealth": 0.85, "core_multiplier": 1.0},
    {"id": 10, "name": "Backdoor Creation",  "category": "persistence", "target": "device",
     "base_reward": 18.0, "severity": 0.90, "stealth": 0.70, "core_multiplier": 1.5},
    # ── Persistence ─────────────────────────────────────────────────────────
    {"id": 11, "name": "Config Tampering",   "category": "persistence", "target": "device",
     "base_reward": 17.0, "severity": 0.85, "stealth": 0.65, "core_multiplier": 1.5},
    {"id": 12, "name": "Shutdown Core Link", "category": "disruption",  "target": "core",
     "base_reward": 20.0, "severity": 1.00, "stealth": 0.50, "core_multiplier": 1.0},
    {"id": 13, "name": "DDoS Core Router",   "category": "disruption",  "target": "core",
     "base_reward": 20.0, "severity": 1.00, "stealth": 0.30, "core_multiplier": 1.0},
    {"id": 14, "name": "Credential Theft",   "category": "recon",       "target": "device",
     "base_reward":  8.0, "severity": 0.40, "stealth": 0.80, "core_multiplier": 1.2},
    {"id": 15, "name": "Lateral Movement",   "category": "persistence", "target": "network",
     "base_reward": 15.0, "severity": 0.80, "stealth": 0.70, "core_multiplier": 1.3},
    {"id": 16, "name": "Firewall Bypass",    "category": "persistence", "target": "firewall",
     "base_reward": 19.0, "severity": 0.95, "stealth": 0.65, "core_multiplier": 1.5},
    {"id": 17, "name": "VLAN Pivot",         "category": "persistence", "target": "switch",
     "base_reward": 16.0, "severity": 0.85, "stealth": 0.70, "core_multiplier": 1.3},
]

# ─────────────────────────────────────────────
# Blue Agent — Defense Action Catalog (18 total)
# ─────────────────────────────────────────────
# Fields:
#   id         : action index (0–17)
#   name       : human-readable label
#   category   : "immediate" | "recovery" | "proactive"
#   counters   : list of Red attack names this action directly neutralises
#   base_reward: base reward on successful counter
BLUE_ACTIONS = [
    # ── Immediate Response ──────────────────────────────────────────────────
    {"id":  0, "name": "Restore Link",           "category": "recovery",
     "counters": ["Shutdown Link", "Shutdown Core Link"],
     "base_reward": 10.0},
    {"id":  1, "name": "Port Isolation",          "category": "immediate",
     "counters": ["MAC Flooding", "VLAN Hopping"],
     "base_reward":  8.0},
    {"id":  2, "name": "MAC Blacklisting",        "category": "immediate",
     "counters": ["MAC Flooding", "ARP Poisoning"],
     "base_reward":  8.0},
    {"id":  3, "name": "Rate Limiting",           "category": "immediate",
     "counters": ["DDoS Simulation", "DDoS Core Router"],
     "base_reward":  9.0},
    {"id":  4, "name": "ICMP Blocking",           "category": "immediate",
     "counters": ["DDoS Simulation", "DDoS Core Router"],
     "base_reward":  7.0},
    {"id":  5, "name": "Traffic Rerouting",       "category": "recovery",
     "counters": ["Shutdown Link", "Shutdown Core Link", "DDoS Core Router"],
     "base_reward": 10.0},
    {"id":  6, "name": "VLAN Reassignment",       "category": "recovery",
     "counters": ["VLAN Hopping", "VLAN Pivot"],
     "base_reward":  9.0},
    {"id":  7, "name": "Config Restoration",      "category": "recovery",
     "counters": ["Config Tampering", "Backdoor Creation"],
     "base_reward": 12.0},
    {"id":  8, "name": "Failover Activation",     "category": "recovery",
     "counters": ["Shutdown Core Link", "DDoS Core Router"],
     "base_reward": 12.0},
    {"id":  9, "name": "Honeypot Deployment",     "category": "proactive",
     "counters": ["Port Scan", "Topology Mapping", "Traffic Sniffing", "Credential Theft"],
     "base_reward": 15.0},
    {"id": 10, "name": "Anomaly Detection Alert", "category": "proactive",
     "counters": ["MAC Flooding", "ARP Poisoning", "DDoS Simulation"],
     "base_reward": 10.0},
    {"id": 11, "name": "Dynamic ACL",             "category": "proactive",
     "counters": ["VLAN Hopping", "Lateral Movement", "Firewall Bypass"],
     "base_reward": 11.0},
    {"id": 12, "name": "Traffic Baselining",      "category": "proactive",
     "counters": ["Traffic Sniffing", "DDoS Simulation"],
     "base_reward":  8.0},
    {"id": 13, "name": "Firewall Rule Update",    "category": "immediate",
     "counters": ["Firewall Bypass", "VLAN Pivot"],
     "base_reward":  9.0},
    {"id": 14, "name": "STP Guard Activation",    "category": "immediate",
     "counters": ["STP Attack"],
     "base_reward": 10.0},
    {"id": 15, "name": "ARP Inspection Enable",   "category": "immediate",
     "counters": ["ARP Poisoning", "MAC Flooding"],
     "base_reward":  9.0},
    {"id": 16, "name": "IDS Alert Response",      "category": "proactive",
     "counters": ["Port Scan", "Topology Mapping", "Lateral Movement"],
     "base_reward": 10.0},
    {"id": 17, "name": "Full Network Lockdown",   "category": "recovery",
     "counters": ["Config Tampering", "Backdoor Creation", "Lateral Movement"],
     "base_reward": 10.0},
]

# ─────────────────────────────────────────────
# RL Dimensions (derived — do not edit manually)
# ─────────────────────────────────────────────
ACTION_DIM = len(RED_ACTIONS)           # 18  (same for Blue — both have 18 choices)
N_METRICS  = 8                          # state groups per interface:
                                        #   status | traffic | errors | cpu |
                                        #   pkt_loss | bcast_ratio | arp_size | anomaly
STATE_DIM  = N_INTERFACES * N_METRICS   # 10 × 8 = 80

# ─────────────────────────────────────────────
# Kill Chain
# ─────────────────────────────────────────────
# Stage progression: 0=none → 1=recon → 2=intrusion → 3=lateral → 4=impact
KILL_CHAIN_RECON_THRESHOLD   = 2      # Red must complete ≥2 recon actions for Stage 1
KILL_CHAIN_COMPLETION_REWARD = 50.0   # Red bonus for finishing all 4 stages
KILL_CHAIN_INTERRUPT_REWARD  = 30.0   # Blue bonus for stopping Red before Stage 4
KILL_CHAIN_ENDS_EPISODE      = True   # Episode ends when kill chain completes (done=True)

# Which Red action categories advance which stage
KILL_CHAIN_STAGE_MAP = {
    1: ("recon",       ["Port Scan", "Topology Mapping", "Traffic Sniffing", "Credential Theft"]),
    2: ("intrusion",   ["ARP Poisoning", "VLAN Hopping", "MAC Flooding", "STP Attack"]),
    3: ("lateral",     ["Lateral Movement", "VLAN Pivot", "Firewall Bypass", "CPU Exhaustion"]),
    4: ("impact",      ["Shutdown Core Link", "DDoS Core Router", "Config Tampering",
                        "Backdoor Creation", "DDoS Simulation"]),
}

# ─────────────────────────────────────────────
# Scenario Types
# ─────────────────────────────────────────────
SCENARIO_TYPES = [
    "normal",               # All links up, Red starts fresh
    "already_compromised",  # One interface already down at episode start
    "high_traffic",         # Network under heavy load when attack begins
    "partial_defense",      # Blue starts with honeypot pre-deployed
    "stealth",              # Red must stay below anomaly threshold to score
]
STEALTH_ANOMALY_THRESHOLD = 0.35    # Anomaly score above this = Red detected

# ─────────────────────────────────────────────
# Reward Config
# ─────────────────────────────────────────────
STEALTH_BONUS             = 5.0     # Red bonus when anomaly stays below threshold
HONEYPOT_PENALTY          = 30.0    # Red penalty if honeypot is triggered
HONEYPOT_BONUS            = 25.0    # Blue bonus for trapping Red in honeypot
CORE_DOWN_STEP_PENALTY    = 2.0     # Blue penalty per step while core is down
WASTED_MOVE_PENALTY       = 2.0     # Blue penalty for acting when no matching attack active
RESPONSE_SPEED_BONUS_MAX  = 5.0     # Blue bonus for 1-step response
PROACTIVE_DETECT_BONUS    = 10.0    # Blue bonus for detecting attack before full impact
CHAIN_INTERRUPT_BONUS     = 30.0    # Blue bonus for stopping kill chain before Stage 4

# ─────────────────────────────────────────────
# DQN Hyperparameters
# ─────────────────────────────────────────────
LEARNING_RATE       = 1e-3
GAMMA               = 0.95       # discount factor
EPSILON_START       = 1.0
EPSILON_MIN         = 0.05
EPSILON_DECAY       = 0.995
MEMORY_SIZE         = 5000
BATCH_SIZE          = 64
TARGET_UPDATE_FREQ  = 10         # sync target network every N training calls
HIDDEN_DIM_1        = 256
HIDDEN_DIM_2        = 128

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
EPISODES            = 300
MAX_STEPS_PER_EP    = 20
STEP_DELAY          = 2.5        # seconds to wait for GNS3 state to propagate
SAVE_INTERVAL       = 50         # save models every N episodes
LOG_DIR             = "logs"
MODEL_DIR           = "models"

# ─────────────────────────────────────────────
# Anomaly Detection (Isolation Forest)
# ─────────────────────────────────────────────
ANOMALY_CONTAMINATION = 0.05     # expected fraction of anomalies

# ─────────────────────────────────────────────
# Predictive Model (LSTM)
# ─────────────────────────────────────────────
LSTM_SEQ_LENGTH     = 60         # look-back window (in data points)
LSTM_HIDDEN_SIZE    = 64
LSTM_NUM_LAYERS     = 2
LSTM_EPOCHS         = 50
LSTM_LEARNING_RATE  = 1e-3
PREDICTION_HORIZON  = 15         # predict N steps ahead

# ─────────────────────────────────────────────
# Backward-compat alias (used by anomaly_detection.py / export_data.py)
# ─────────────────────────────────────────────
ACTIONS_CONFIG = INTERFACE_ACTIONS
