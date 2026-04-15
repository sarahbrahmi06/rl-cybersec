import requests

class GNS3Connector:
    """
    Sarah's GNS3 Connector for v3 REST API.
    Houssem: This class handles the physical link toggling in the topology.
    """
    def __init__(self):
        # API Configuration
        self.base_url = "http://127.0.0.1:3080/v3"
        self.project_id = "1e81b342-9ba3-40a4-b48c-12fdd94787bc"
        self.username = "admin"
        self.password = "project2026"
        # Access token obtained via Sarah's authentication test
        self.token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTc3NjAyMTI5Nn0.As8kAqdssmTTZ3Z010Tgg6MBvntRM2NQtcKpSUFALM8"

    def get_headers(self):
        """Standard headers for GNS3 v3 (Bearer Token + JSON)"""
        return {
            "Authorization": f"Bearer {self.token}", 
            "Content-Type": "application/json"
        }

    def _set_link_state(self, node_name, adapter, should_suspend):
        """
        CORE LOGIC: In GNS3 v3, links are cut by updating the 'suspend' boolean property.
        Houssem: should_suspend=True cuts the link, should_suspend=False restores it.
        """
        try:
            # 1. Resolve Node Name to Node ID
            nodes = requests.get(f"{self.base_url}/projects/{self.project_id}/nodes", headers=self.get_headers()).json()
            node_id = next((n["node_id"] for n in nodes if n["name"] == node_name), None)
            if not node_id: return False

            # 2. Search for the specific link connected to that Node and Adapter/Port
            links = requests.get(f"{self.base_url}/projects/{self.project_id}/links", headers=self.get_headers()).json()
            link_id = None
            for link in links:
                if any(n["node_id"] == node_id and n["adapter_number"] == adapter for n in link["nodes"]):
                    link_id = link["link_id"]
                    break
            
            if not link_id: return False

            # 3. Use PUT to modify the link object state in the API
            url = f"{self.base_url}/projects/{self.project_id}/links/{link_id}"
            r = requests.put(url, headers=self.get_headers(), json={"suspend": should_suspend})
            
            # Returns True if GNS3 server accepts the state change (200/201/204)
            return r.status_code in [200, 201, 204]
        except Exception as e:
            print(f"Connector Error: {e}")
            return False

    def shutdown_interface(self, name, port): 
        """Utility for Red Agent to disable a link"""
        return self._set_link_state(name, port, True)
        
    def restore_interface(self, name, port): 
        """Utility for Blue Agent to enable a link"""
        return self._set_link_state(name, port, False)