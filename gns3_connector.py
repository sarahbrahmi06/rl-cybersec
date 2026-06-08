"""
GNS3 v3 API Connector (Extended)
==================================
Handles link suspension/restoration AND node-level operations
(CPU exhaustion simulation, config retrieval, node pause/resume).
"""

import requests
from config import GNS3_BASE_URL, GNS3_PROJECT_ID, GNS3_TOKEN


class GNS3Connector:
    """
    GNS3 v3 API Connector.

    Provides:
    - Link-level: shutdown_interface / restore_interface  (existing)
    - Node-level: suspend_node / resume_node              (new)
    - Topology:   get_node_list / get_link_list           (new)
    - Config:     get_node_config / push_node_config      (new, Telnet-based — simulated)
    """

    def __init__(self, base_url=None, project_id=None, token=None):
        self.base_url   = base_url   or GNS3_BASE_URL
        self.project_id = project_id or GNS3_PROJECT_ID
        self.token      = token      or GNS3_TOKEN

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type":  "application/json",
        }

    # ── Link-level (existing) ────────────────────────────

    def shutdown_interface(self, node_name: str, adapter: int) -> bool:
        """Suspend (cut) a link attached to *node_name* on *adapter*."""
        return self._set_link_state(node_name, adapter, True)

    def restore_interface(self, node_name: str, adapter: int) -> bool:
        """Restore a suspended link attached to *node_name* on *adapter*."""
        return self._set_link_state(node_name, adapter, False)

    # ── Topology queries (new) ───────────────────────────

    def get_node_list(self) -> list:
        """
        Return the full list of node objects for this project.

        Returns
        -------
        list[dict]
            Each dict is the raw GNS3 node object (keys: node_id, name, status, …).
            Returns an empty list on error.
        """
        try:
            resp = requests.get(
                f"{self.base_url}/projects/{self.project_id}/nodes",
                headers=self.get_headers(), timeout=10,
            )
            return resp.json() if resp.status_code == 200 else []
        except Exception as e:
            print(f"[GNS3] get_node_list error: {e}")
            return []

    def get_link_list(self) -> list:
        """
        Return the full list of link objects for this project.

        Returns
        -------
        list[dict]
            Each dict is the raw GNS3 link object (keys: link_id, nodes, suspend, …).
            Returns an empty list on error.
        """
        try:
            resp = requests.get(
                f"{self.base_url}/projects/{self.project_id}/links",
                headers=self.get_headers(), timeout=10,
            )
            return resp.json() if resp.status_code == 200 else []
        except Exception as e:
            print(f"[GNS3] get_link_list error: {e}")
            return []

    def get_node_id(self, node_name: str) -> str | None:
        """Resolve a node name to its GNS3 node_id (None if not found)."""
        nodes = self.get_node_list()
        return next((n["node_id"] for n in nodes if n["name"] == node_name), None)

    # ── Node-level control (new) ─────────────────────────

    def suspend_node(self, node_name: str) -> bool:
        """
        Pause / suspend a GNS3 node to simulate CPU exhaustion or DDoS impact.
        Uses the GNS3 v3 node suspend endpoint.

        Returns True on success, False on failure.
        """
        node_id = self.get_node_id(node_name)
        if not node_id:
            print(f"[GNS3] suspend_node: '{node_name}' not found")
            return False
        try:
            url  = f"{self.base_url}/projects/{self.project_id}/nodes/{node_id}/suspend"
            resp = requests.post(url, headers=self.get_headers(), timeout=10)
            ok   = resp.status_code in [200, 201, 204]
            if ok:
                print(f"[GNS3] Suspended node: {node_name}")
            return ok
        except Exception as e:
            print(f"[GNS3] suspend_node error: {e}")
            return False

    def resume_node(self, node_name: str) -> bool:
        """
        Resume a previously suspended GNS3 node.

        Returns True on success, False on failure.
        """
        node_id = self.get_node_id(node_name)
        if not node_id:
            print(f"[GNS3] resume_node: '{node_name}' not found")
            return False
        try:
            url  = f"{self.base_url}/projects/{self.project_id}/nodes/{node_id}/start"
            resp = requests.post(url, headers=self.get_headers(), timeout=10)
            ok   = resp.status_code in [200, 201, 204]
            if ok:
                print(f"[GNS3] Resumed node: {node_name}")
            return ok
        except Exception as e:
            print(f"[GNS3] resume_node error: {e}")
            return False

    def stop_node(self, node_name: str) -> bool:
        """Stop a GNS3 node (harder than suspend — simulates power-off / DDoS kill)."""
        node_id = self.get_node_id(node_name)
        if not node_id:
            return False
        try:
            url  = f"{self.base_url}/projects/{self.project_id}/nodes/{node_id}/stop"
            resp = requests.post(url, headers=self.get_headers(), timeout=10)
            return resp.status_code in [200, 201, 204]
        except Exception as e:
            print(f"[GNS3] stop_node error: {e}")
            return False

    def start_node(self, node_name: str) -> bool:
        """Start a stopped GNS3 node."""
        node_id = self.get_node_id(node_name)
        if not node_id:
            return False
        try:
            url  = f"{self.base_url}/projects/{self.project_id}/nodes/{node_id}/start"
            resp = requests.post(url, headers=self.get_headers(), timeout=10)
            return resp.status_code in [200, 201, 204]
        except Exception as e:
            print(f"[GNS3] start_node error: {e}")
            return False

    # ── Config (simulated — requires Telnet/SSH in real deployment) ──────────

    def get_node_config(self, node_name: str) -> str:
        """
        Retrieve the running configuration of a GNS3 node.

        NOTE: In real GNS3 this requires a Telnet connection to the console.
        This method returns a placeholder config string. Replace with
        a real Telnet implementation when deploying against a live topology.

        Returns
        -------
        str
            The running config (simulated placeholder).
        """
        # Placeholder — real implementation would open a Telnet session
        # to the node's console port and send "show running-config"
        return f"! Simulated running-config for {node_name}\ninterface GigabitEthernet0/0\n no shutdown\n!\n"

    def push_node_config(self, node_name: str, config_commands: str) -> bool:
        """
        Push configuration commands to a GNS3 node via console.

        NOTE: Simulated. Replace with a real Telnet/SSH implementation.

        Parameters
        ----------
        node_name       : str  — GNS3 node name
        config_commands : str  — Newline-separated IOS/NX-OS commands

        Returns
        -------
        bool — True (simulated success)
        """
        print(f"[GNS3] push_node_config({node_name}): simulated. "
              f"Commands:\n{config_commands}")
        return True

    def restore_node_config(self, node_name: str) -> bool:
        """
        Restore a node's configuration to a known-good baseline.
        Simulated — push the baseline config back.
        """
        baseline = self.get_node_config(node_name)  # Simulated baseline
        return self.push_node_config(node_name, baseline)

    # ── Internal helpers ─────────────────────────────────

    def _set_link_state(self, node_name: str, adapter: int, should_suspend: bool) -> bool:
        from config import TELEMETRY_MODE
        if TELEMETRY_MODE == "synthetic":
            return True
            
        try:
            # Fetch all nodes
            nodes = requests.get(
                f"{self.base_url}/projects/{self.project_id}/nodes",
                headers=self.get_headers(), timeout=10,
            ).json()
            node_id = next(
                (n["node_id"] for n in nodes if n["name"] == node_name), None
            )
            if not node_id:
                print(f"[GNS3] Node '{node_name}' not found")
                return False

            # Find the link connected to that adapter
            links = requests.get(
                f"{self.base_url}/projects/{self.project_id}/links",
                headers=self.get_headers(), timeout=10,
            ).json()
            link_id = None
            for link in links:
                if any(
                    n["node_id"] == node_id and n["adapter_number"] == adapter
                    for n in link["nodes"]
                ):
                    link_id = link["link_id"]
                    break

            if not link_id:
                print(f"[GNS3] No link found for {node_name} adapter {adapter}")
                return False

            # Update link state
            url      = f"{self.base_url}/projects/{self.project_id}/links/{link_id}"
            response = requests.put(
                url, headers=self.get_headers(),
                json={"suspend": should_suspend}, timeout=10,
            )
            return response.status_code in [200, 201, 204]

        except requests.exceptions.ConnectionError:
            print(f"[GNS3 Connector Error] Cannot reach GNS3 server at {self.base_url}")
            return False
        except requests.exceptions.Timeout:
            print("[GNS3 Connector Error] Request timed out")
            return False
        except Exception as e:
            print(f"[GNS3 Connector Error] {e}")
            return False