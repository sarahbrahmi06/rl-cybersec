from influxdb_client import InfluxDBClient
import numpy as np

class StateCollector:
    def __init__(self):
        self.client = InfluxDBClient(
            url="http://10.202.91.55:8086",
            token="project2026token",
            org="project"
        )
        self.query_api = self.client.query_api()

        # 10 monitored interfaces → state vector size = 30
        self.monitored_interfaces = [
            "GigabitEthernet0/0", "GigabitEthernet0/1", "GigabitEthernet0/2", "GigabitEthernet0/3",
            "GigabitEthernet1/0", "GigabitEthernet1/1", "GigabitEthernet1/2", "GigabitEthernet1/3",
            "GigabitEthernet0/0", "GigabitEthernet0/2"   # duplicates are okay for now, we can clean later
        ]

    def get_interface_status(self):
        query = '''
        from(bucket:"network")
            |> range(start: -2m)
            |> filter(fn: (r) => r._measurement == "interface")
            |> filter(fn: (r) => r._field == "status")
            |> last()
        '''
        try:
            result = self.query_api.query(query)
            status = {}
            for table in result:
                for record in table.records:
                    name = record.values.get("name", "unknown")
                    status[name] = int(record.get_value()) if record.get_value() is not None else 1
            return status
        except Exception as e:
            print(f"[-] Error reading interface status: {e}")
            return {}

    def get_traffic_rate(self):
        query = '''
        from(bucket:"network")
            |> range(start: -2m)
            |> filter(fn: (r) => r._measurement == "interface")
            |> filter(fn: (r) => r._field == "in_octets")
            |> last()
        '''
        try:
            result = self.query_api.query(query)
            traffic = {}
            for table in result:
                for record in table.records:
                    name = record.values.get("name", "unknown")
                    traffic[name] = record.get_value() or 0
            return traffic
        except Exception as e:
            print(f"[-] Error reading traffic rate: {e}")
            return {}

    def get_error_count(self):
        query = '''
        from(bucket:"network")
            |> range(start: -2m)
            |> filter(fn: (r) => r._measurement == "interface")
            |> filter(fn: (r) => r._field == "in_errors")
            |> last()
        '''
        try:
            result = self.query_api.query(query)
            errors = {}
            for table in result:
                for record in table.records:
                    name = record.values.get("name", "unknown")
                    errors[name] = record.get_value() or 0
            return errors
        except Exception as e:
            print(f"[-] Error reading error counts: {e}")
            return {}

    def get_full_state(self):
        return {
            "interface_status": self.get_interface_status(),
            "traffic_rate": self.get_traffic_rate(),
            "error_count": self.get_error_count()
        }

    def get_state_vector(self):
        state = self.get_full_state()
        
        status_values = np.array([
            1.0 if state["interface_status"].get(iface, 1) == 1 else 0.0
            for iface in self.monitored_interfaces
        ], dtype=np.float32)
        
        traffic_values = np.array([
            np.log1p(state["traffic_rate"].get(iface, 0) / 1000.0)
            for iface in self.monitored_interfaces
        ], dtype=np.float32)
        
        error_values = np.array([
            min(state["error_count"].get(iface, 0) / 100.0, 10.0)
            for iface in self.monitored_interfaces
        ], dtype=np.float32)
        
        return np.concatenate([status_values, traffic_values, error_values])


# ==================== TEST BLOCK ====================
if __name__ == "__main__":
    print("[*] Reading network state from InfluxDB...")
    collector = StateCollector()
    
    state = collector.get_full_state()
    vector = collector.get_state_vector()
    
    print(f"\n[+] Interface Status: {state['interface_status']}")
    print(f"[+] Traffic Rate: {state['traffic_rate']}")
    print(f"[+] Error Counts: {state['error_count']}")
    print(f"\n[+] State Vector Shape: {vector.shape}")
    print(f"[+] State Vector: {vector}")