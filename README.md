#  Cyber-Arena: Autonomous Cybersecurity Simulation Platform

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![GNS3](https://img.shields.io/badge/GNS3-3.0+-green.svg)](https://gns3.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Real-time cybersecurity simulation where AI agents learn to attack and defend an enterprise network.**

Cyber-Arena combines GNS3 network emulation, live telemetry (Telegraf → InfluxDB → Grafana), and Deep Q-Network agents to create a closed-loop autonomous cybersecurity training platform.

---

##  Key Features

| Feature | Description |
|---------|-------------|
| ** Complete Enterprise Topology** | 8 PCs, 5 VLANs, 2 Cisco routers, 2 L3 switches, 4 L2 switches, Alpine firewall, victim host |
| ** RL Agents** | Red (attacker) and Blue (defender) DQN agents with 80-dim state, 18 actions each |
| ** Cyber Kill Chain** | 4 stages: Reconnaissance → Intrusion → Lateral Movement → Impact |
| ** Live Telemetry** | SNMP → Telegraf → InfluxDB → Grafana pipeline with real-time dashboards |
| ** Honeypot System** | Blue deploys honeypots to trap Red's recon attacks |
| ** Firewall Management** | Blue updates firewall rules to counter bypass attempts |
| ** Ansible Automation** | Daily config backups for 6 devices with Prometheus monitoring |
| ** ZeroTier Integration** | Remote monitoring stack access |

---

##  Project Infrastructure

<img width="4300" height="2900" alt="Copy of Infrastructure_architecture" src="https://github.com/user-attachments/assets/33b03310-820d-4b46-8d2d-20f0ac7a9d95" />

##  Project Structure

<img width="1253" height="592" alt="image" src="https://github.com/user-attachments/assets/ea23a158-f64d-426d-b972-7763226c7cb1" />
<img width="1264" height="777" alt="Screenshot From 2026-05-27 21-19-26" src="https://github.com/user-attachments/assets/61af862f-8aa1-4f2f-a1cc-0d747d18d147" />

<img width="1276" height="756" alt="Screenshot From 2026-05-29 19-07-16" src="https://github.com/user-attachments/assets/ea22bcd5-7001-4cd1-b66f-6762155041cd" />

```bash
rl-cybersec/
├── config.py                 # All hyperparameters
├── environment.py            # Gymnasium RL environment
├── red_agent.py              # DQN attacker
├── blue_agent.py             # DQN defender with PER
├── attack_executor.py        # Executes Red attacks
├── defense_executor.py       # Executes Blue defenses
├── state_collector.py        # Builds 80-dim state
├── gns3_connector.py         # GNS3 v3 API client
├── train.py                  # Main training loop
├── anomaly_detection.py      # Isolation Forest
├── export_data.py            # Export InfluxDB data
├── requirements.txt          # Python dependencies




