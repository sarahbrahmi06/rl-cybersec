import time
from gns3_connector import GNS3Connector
from state_collector import StateCollector

class NetworkEnvironment:
    """
    Sarah's CyberSec Environment Wrapper.
    Houssem: This acts as the 'Gym' environment for the Red and Blue agents.
    It bridges the GNS3 Infrastructure with your RL Training loop.
    """
    def __init__(self):
        # Initialize the GNS3 Controller and the InfluxDB Telemetry Collector
        self.gns3 = GNS3Connector()
        self.collector = StateCollector() 
        
        # ACTION SPACE MAPPING
        # Defines which index (0-7) corresponds to which physical link.
        # Houssem: Note that port 2 on L3 switches represents the Core Infrastructure.
        self.actions_config = [
            ("Switch1", 0), ("Switch2", 0), ("Switch3", 0), ("Switch4", 0),
            ("SwitchL31", 0), ("SwitchL31", 2), ("SwitchL32", 0), ("SwitchL32", 2),
        ]
        self.action_count = len(self.actions_config)

    def get_state(self):
        """
        Houssem: Returns the 30-dimension telemetry vector from InfluxDB.
        This includes traffic throughput, error rates, and latency metrics.
        """
        return self.collector.get_state_vector()

    def red_step(self, index):
        """
        RED AGENT STEP: Disruption
        - Rewards: +20.0 for Core (L3/Port 2), +10.0 for Edge.
        - Penalty: -1.0 if the action fails or the link is already down.
        """
        node, port = self.actions_config[index]
        success = self.gns3.shutdown_interface(node, port)
        
        # Brief sleep to allow the network changes to reflect in InfluxDB telemetry
        time.sleep(1.5) 
        
        if success:
            # Reward logic based on importance of the node
            reward = 20.0 if "L3" in node and port == 2 else 10.0
            print(f"[RED ACTION] Success: {node} port {port} DOWN | Reward: {reward}")
            return self.get_state(), reward, False
        
        # If the API returns False (e.g., link already suspended), return penalty
        return self.get_state(), -1.0, False

    def blue_step(self, index):
        """
        BLUE AGENT STEP: Resilience/Restoration
        - Rewards: +20.0 for Core, +10.0 for Edge.
        - Penalty: -1.0 if the action fails or the link is already up.
        """
        node, port = self.actions_config[index]
        success = self.gns3.restore_interface(node, port)
        
        time.sleep(1.5)
        
        if success:
            reward = 20.0 if "L3" in node and port == 2 else 10.0
            print(f"[BLUE ACTION] Success: {node} port {port} RESTORED | Reward: {reward}")
            return self.get_state(), reward, False
            
        return self.get_state(), -1.0, False

    def reset(self):
        """
        Houssem: Resets project to baseline (All Links UP).
        Call this at the beginning of every training episode.
        """
        print("[ENVIRONMENT] Baseline Reset Triggered: Restoring all links...")
        for node, port in self.actions_config:
            self.gns3.restore_interface(node, port)
        time.sleep(2)
        return self.get_state()

# =================================================================
# EXECUTION BLOCK 
# =================================================================
if __name__ == "__main__":
    # 1. Instantiate the environment
    env = NetworkEnvironment()
    
    # 2. Reset the system to a known good state
    initial_vector = env.reset()
    print(f"[+] System Ready. Telemetry Vector size: {len(initial_vector)}")

    # 3. Test Attack on Core (Action Index 5)
    print("\n--- TEST: RED ATTACK ON CORE ---")
    state, rew, done = env.red_step(5)
    print(f"Returned Reward: {rew}")

    time.sleep(1)

    # 4. Test Restoration on Core (Action Index 5)
    print("\n--- TEST: BLUE RESTORE ON CORE ---")
    state, rew, done = env.blue_step(5)
    print(f"Returned Reward: {rew}")

    print("\n[✔] Sarah's Environment Test Complete.")