"""
State Collector (Extended — 80-dimensional)
============================================
Collects real-time network telemetry from InfluxDB and builds an 80-dim
observation vector for the RL agents.

State vector layout (N = 10 interfaces):
  [0   : N]   — interface up/down status      (1.0 = up, 0.0 = down)
  [N   : 2N]  — log-scaled traffic rate       (in_octets)
  [2N  : 3N]  — normalised error count        (in_errors)
  [3N  : 4N]  — CPU usage per device          (0–1 fraction)
  [4N  : 5N]  — packet loss percentage        (0–1 fraction)
  [5N  : 6N]  — broadcast traffic ratio       (0–1; high = MAC flood indicator)
  [6N  : 7N]  — ARP table size (normalised)   (high = ARP attack indicator)
  [7N  : 8N]  — anomaly score                 (0–1; how far from normal baseline)
"""

import numpy as np
from influxdb_client import InfluxDBClient

from config import (
    INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG,
    INFLUXDB_BUCKET, MONITORED_INTERFACES, N_INTERFACES,
    STATE_DIM,
)


class StateCollector:
    """
    Collects real-time network telemetry from InfluxDB.

    Produces a state vector of shape (STATE_DIM,) = (80,) where each
    of the 8 metric groups spans N_INTERFACES = 10 values.
    """

    def __init__(self):
        self.client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG,
        )
        self.query_api = self.client.query_api()
        self.monitored_interfaces = list(MONITORED_INTERFACES)

    # ── Individual metric queries ────────────────────────

    def get_interface_status(self) -> dict:
        """Return {iface: status_int} — 1 = up, 2 = down."""
        query = f'''
        from(bucket:"{INFLUXDB_BUCKET}")
            |> range(start: -2m)
            |> filter(fn: (r) => r._measurement == "interface")
            |> filter(fn: (r) => r._field == "status")
            |> last()
        '''
        return self._query_dict(query, default=1)

    def get_traffic_rate(self) -> dict:
        """Return {iface: in_octets} from the latest SNMP poll."""
        query = f'''
        from(bucket:"{INFLUXDB_BUCKET}")
            |> range(start: -2m)
            |> filter(fn: (r) => r._measurement == "interface")
            |> filter(fn: (r) => r._field == "in_octets")
            |> last()
        '''
        return self._query_dict(query, default=0)

    def get_error_count(self) -> dict:
        """Return {iface: in_errors} from the latest SNMP poll."""
        query = f'''
        from(bucket:"{INFLUXDB_BUCKET}")
            |> range(start: -2m)
            |> filter(fn: (r) => r._measurement == "interface")
            |> filter(fn: (r) => r._field == "in_errors")
            |> last()
        '''
        return self._query_dict(query, default=0)

    def get_cpu_usage(self) -> dict:
        """
        Return {iface: cpu_fraction} — CPU usage per device (0.0–1.0).

        Queries the "device" measurement for cpu_usage field.
        Falls back to 0.1 (10 % idle baseline) if not available.
        """
        query = f'''
        from(bucket:"{INFLUXDB_BUCKET}")
            |> range(start: -2m)
            |> filter(fn: (r) => r._measurement == "device")
            |> filter(fn: (r) => r._field == "cpu_usage")
            |> last()
        '''
        raw = self._query_dict(query, default=0.1, name_field="interface")
        # CPU may be keyed by device name; map to interface names
        return {iface: raw.get(iface, 0.1) for iface in self.monitored_interfaces}

    def get_packet_loss(self) -> dict:
        """
        Return {iface: loss_fraction} — estimated from error/traffic ratio.
        Uses in_discards field if available, otherwise estimates from in_errors.
        """
        query = f'''
        from(bucket:"{INFLUXDB_BUCKET}")
            |> range(start: -2m)
            |> filter(fn: (r) => r._measurement == "interface")
            |> filter(fn: (r) => r._field == "in_discards")
            |> last()
        '''
        discards = self._query_dict(query, default=0)
        traffic  = self.get_traffic_rate()

        result = {}
        for iface in self.monitored_interfaces:
            d = discards.get(iface, 0)
            t = max(traffic.get(iface, 1), 1)          # avoid div/0
            result[iface] = min(float(d) / float(t), 1.0)
        return result

    def get_broadcast_ratio(self) -> dict:
        """
        Return {iface: broadcast_ratio} — fraction of traffic that is broadcast.
        High values (> 0.5) indicate a MAC flooding / broadcast storm.

        Queries broadcast_pkts field from InfluxDB.
        Falls back to 0.0 if not available.
        """
        query = f'''
        from(bucket:"{INFLUXDB_BUCKET}")
            |> range(start: -2m)
            |> filter(fn: (r) => r._measurement == "interface")
            |> filter(fn: (r) => r._field == "broadcast_pkts")
            |> last()
        '''
        bcast   = self._query_dict(query, default=0)
        traffic = self.get_traffic_rate()

        result = {}
        for iface in self.monitored_interfaces:
            b = bcast.get(iface, 0)
            t = max(traffic.get(iface, 1), 1)
            result[iface] = min(float(b) / float(t), 1.0)
        return result

    def get_arp_table_size(self) -> dict:
        """
        Return {iface: normalised_arp_size} — abnormal growth indicates ARP attack.
        Normalised to [0, 1] assuming 200 entries is maximum expected.

        Queries arp_entries field from InfluxDB.
        Falls back to a baseline of 0.05 (≈10 entries).
        """
        query = f'''
        from(bucket:"{INFLUXDB_BUCKET}")
            |> range(start: -2m)
            |> filter(fn: (r) => r._measurement == "device")
            |> filter(fn: (r) => r._field == "arp_entries")
            |> last()
        '''
        raw = self._query_dict(query, default=10, name_field="interface")
        ARP_MAX = 200.0
        return {
            iface: min(raw.get(iface, 10) / ARP_MAX, 1.0)
            for iface in self.monitored_interfaces
        }

    def get_anomaly_scores(self, baseline_traffic: dict = None) -> dict:
        """
        Return {iface: anomaly_score ∈ [0, 1]}.

        Computed as the normalised z-score of current traffic vs a rolling
        baseline. If no baseline is provided, uses a rough heuristic
        (deviation from log-scaled mean).

        Parameters
        ----------
        baseline_traffic : dict | None
            Pre-computed {iface: mean_octets} baseline. If None, uses the
            current traffic values and marks deviation from their mean.
        """
        traffic = self.get_traffic_rate()
        values  = np.array([traffic.get(iface, 0) for iface in self.monitored_interfaces],
                           dtype=np.float64)

        if baseline_traffic is not None:
            baseline = np.array(
                [baseline_traffic.get(iface, 0) for iface in self.monitored_interfaces],
                dtype=np.float64,
            )
        else:
            # Use mean of current observations as a rough baseline
            baseline = np.full_like(values, values.mean() if values.mean() > 0 else 1.0)

        diffs  = np.abs(values - baseline)
        norm   = baseline + 1e-6
        scores = np.clip(diffs / norm, 0.0, 1.0).astype(np.float32)
        return {iface: float(scores[i]) for i, iface in enumerate(self.monitored_interfaces)}

    # ── Composite state ──────────────────────────────────

    def get_full_state(self) -> dict:
        """Return a dict with all 8 metric categories."""
        return {
            "interface_status":  self.get_interface_status(),
            "traffic_rate":      self.get_traffic_rate(),
            "error_count":       self.get_error_count(),
            "cpu_usage":         self.get_cpu_usage(),
            "packet_loss":       self.get_packet_loss(),
            "broadcast_ratio":   self.get_broadcast_ratio(),
            "arp_table_size":    self.get_arp_table_size(),
            "anomaly_score":     self.get_anomaly_scores(),
        }

    def get_state_vector(self) -> np.ndarray:
        """
        Build a flat numpy vector suitable as an 80-dim RL observation.

        Layout:
          [status × N | traffic × N | errors × N | cpu × N |
           pkt_loss × N | bcast_ratio × N | arp_size × N | anomaly × N]
        """
        state = self.get_full_state()
        ifaces = self.monitored_interfaces

        def _vec(metric_dict, scale_fn=None, default=0.0):
            vals = [metric_dict.get(iface, default) for iface in ifaces]
            if scale_fn:
                vals = [scale_fn(v) for v in vals]
            return np.array(vals, dtype=np.float32)

        status_values  = _vec(state["interface_status"],
                               scale_fn=lambda v: 1.0 if v == 1 else 0.0)
        traffic_values = _vec(state["traffic_rate"],
                               scale_fn=lambda v: float(np.log1p(v / 1000.0)))
        error_values   = _vec(state["error_count"],
                               scale_fn=lambda v: min(v / 100.0, 10.0))
        cpu_values     = _vec(state["cpu_usage"])
        pkt_loss_vals  = _vec(state["packet_loss"])
        bcast_values   = _vec(state["broadcast_ratio"])
        arp_values     = _vec(state["arp_table_size"])
        anomaly_values = _vec(state["anomaly_score"])

        vec = np.concatenate([
            status_values,
            traffic_values,
            error_values,
            cpu_values,
            pkt_loss_vals,
            bcast_values,
            arp_values,
            anomaly_values,
        ])
        assert vec.shape == (STATE_DIM,), f"State dim mismatch: {vec.shape} != ({STATE_DIM},)"
        return vec

    # ── Internal helpers ─────────────────────────────────

    def _query_dict(self, query: str, default=0, name_field: str = "name") -> dict:
        """Run a Flux query and return {interface_name: value} dict."""
        try:
            result = self.query_api.query(query)
            out = {}
            for table in result:
                for record in table.records:
                    name  = record.values.get(name_field, "unknown")
                    value = record.get_value()
                    out[name] = value if value is not None else default
            return out
        except Exception as e:
            print(f"[-] InfluxDB query error: {e}")
            return {}


# ==================== TEST BLOCK ====================
if __name__ == "__main__":
    print("[*] Reading 80-dim network state from InfluxDB...")
    collector = StateCollector()

    state  = collector.get_full_state()
    vector = collector.get_state_vector()

    for metric, data in state.items():
        print(f"\n[+] {metric}: {data}")

    print(f"\n[+] State Vector Shape : {vector.shape}")
    print(f"[+] State Vector (first 30): {vector[:30]}")