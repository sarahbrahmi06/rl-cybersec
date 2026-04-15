import requests

class GNS3Connector:
    """
    GNS3 v3 API Connector
    Handles suspending and restoring links in the topology.
    """

    def __init__(self):
        self.base_url = "http://127.0.0.1:3080/v3"
        self.project_id = "1e81b342-9ba3-40a4-b48c-12fdd94787bc"
        self.token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTc3NjAyMTI5Nn0.As8kAqdssmTTZ3Z010Tgg6MBvntRM2NQtcKpSUFALM8"

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    def shutdown_interface(self, node_name, adapter):
        """Suspend (cut) a link"""
        return self._set_link_state(node_name, adapter, True)

    def restore_interface(self, node_name, adapter):
        """Restore a suspended link"""
        return self._set_link_state(node_name, adapter, False)

    def _set_link_state(self, node_name, adapter, should_suspend):
        try:
            # Get nodes
            nodes = requests.get(f"{self.base_url}/projects/{self.project_id}/nodes", 
                               headers=self.get_headers()).json()
            node_id = next((n["node_id"] for n in nodes if n["name"] == node_name), None)
            if not node_id:
                return False

            # Find the link
            links = requests.get(f"{self.base_url}/projects/{self.project_id}/links", 
                               headers=self.get_headers()).json()
            link_id = None
            for link in links:
                if any(n["node_id"] == node_id and n["adapter_number"] == adapter for n in link["nodes"]):
                    link_id = link["link_id"]
                    break

            if not link_id:
                return False

            # Update link state
            url = f"{self.base_url}/projects/{self.project_id}/links/{link_id}"
            response = requests.put(url, headers=self.get_headers(), 
                                  json={"suspend": should_suspend})
            return response.status_code in [200, 201, 204]

        except Exception as e:
            print(f"[GNS3 Connector Error] {e}")
            return False