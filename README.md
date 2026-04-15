# Autonomous Red-vs-Blue Cybersecurity System

**Reinforcement Learning on a GNS3 Network Digital Twin**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![GNS3](https://img.shields.io/badge/GNS3-4A4A4A?style=for-the-badge&logo=gns3&logoColor=white)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement%20Learning-FF6F00?style=for-the-badge)

---

## Overview

This project implements an **autonomous Red-vs-Blue cybersecurity framework** using Reinforcement Learning (RL). Two intelligent agents compete in a realistic network environment:

- **Red Agent**: Learns offensive strategies to disrupt network connectivity.
- **Blue Agent**: Learns defensive strategies to detect attacks and restore services.

Both agents improve continuously through **self-play training** inside a live **Network Digital Twin** built with GNS3.

The system demonstrates how modern AI can create adaptive, self-evolving security mechanisms that go beyond traditional static defenses.

---

## Key Features

- Realistic 17-node 3-tier hierarchical network simulated in GNS3
- Real-time Network Digital Twin with SNMP telemetry (Telegraf + InfluxDB)
- Live monitoring and visualization using Grafana
- Custom Gym-like RL environment for seamless agent training
- Competitive self-play between Red and Blue agents
- Modular architecture with clear separation of infrastructure and AI layers

---

## Architecture

The system is built in three layers:

1. **Network Simulation Layer** : GNS3 3.0.6 with 2 routers, 2 Layer-3 switches, 4 access switches, and 8 end hosts
2. **Monitoring & Digital Twin Layer** : Telegraf, InfluxDB, Grafana, and NetBox for real-time metrics and network documentation
3. **AI Intelligence Layer** : Custom RL environment + agents powered by stable-baselines3 and Gymnasium

All components are connected securely via ZeroTier VPN for team collaboration.

---

## Repository Structure

```bash
rl-cybersec/
├── environment/           # Core RL Environment (Sarah)
│   ├── gns3_connector.py
│   ├── state_collector.py
│   └── environment.py
├── agents/                # RL Agents (Houssem)
│   ├── red_agent.py
│   ├── blue_agent.py
│   └── train.py
├── monitoring/            # Grafana dashboards & configs
├── docs/
├── requirements.txt
└── README.md
