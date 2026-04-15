import time
from .gns3_connector import GNS3Connector
from .state_collector import StateCollector

class NetworkEnvironment:
    """
    Main RL Environment for Red vs Blue Agents
    Sarah Brahmi - Infrastructure Layer
    """

    def __init__(self):
        print("[*] Initializing RL Environment...")
        self.gns3 = GNS3Connector()
        self.collector = StateCollector()

        self.actions_config = [
            ("Switch1", 0), ("Switch2", 0), ("Switch3", 0), ("Switch4", 0),
            ("SwitchL31", 0), ("SwitchL31", 2),   # Core uplink
            ("SwitchL32", 0), ("SwitchL32", 2),   # Core uplink
        ]
        self.action_count = len(self.actions_config)

        self.interfaces_down = set()
        print(f"[+] Environment ready | Actions: {self.action_count} | State size: 30")

    def get_state(self):
        return self.collector.get_state_vector()

    def count_down_interfaces(self):
        state_dict = self.collector.get_full_state()
        return sum(1 for v in state_dict.get("interface_status", {}).values() if v == 2)

    def red_step(self, index):
        """Red Agent - Attack"""
        if index >= self.action_count:
            return self.get_state(), -1.0, False

        node, port = self.actions_config[index]
        success = self.gns3.shutdown_interface(node, port)
        time.sleep(2.5)

        down_count = self.count_down_interfaces()

        if not success:
            reward = -1.0
        elif "L3" in node and port == 2:        # Core uplink
            reward = 20.0
        elif down_count > 0:
            reward = 10.0
        else:
            reward = -1.0

        if success and down_count > 0:
            self.interfaces_down.add((node, port))

        return self.get_state(), reward, False

    def blue_step(self, index):
        """Blue Agent - Defense"""
        if index >= self.action_count:
            return self.get_state(), -1.0, False

        node, port = self.actions_config[index]
        success = self.gns3.restore_interface(node, port)
        time.sleep(2.5)

        down_count = self.count_down_interfaces()

        if not success:
            reward = -1.0
        elif down_count == 0:                   # Full recovery
            reward = 10.0
        else:
            reward = 5.0

        if success:
            self.interfaces_down.discard((node, port))

        done = (down_count == 0)
        return self.get_state(), reward, done

    def reset(self):
        """Reset all links to UP"""
        print("[*] Resetting environment — restoring all links...")
        for node, port in self.actions_config:
            self.gns3.restore_interface(node, port)

        self.interfaces_down.clear()
        time.sleep(3)
        print("[+] Reset complete.")
        return self.get_state()


# Test
if __name__ == "__main__":
    env = NetworkEnvironment()
    state = env.reset()

    print("\n--- Red Test (Core Uplink) ---")
    state, rew, done = env.red_step(5)
    print(f"Red Reward: {rew}")

    print("\n--- Blue Test (Restore) ---")
    state, rew, done = env.blue_step(5)
    print(f"Blue Reward: {rew} | Done: {done}")