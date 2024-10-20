import pickle
import os
import random


def load_client_profile(file_path):
    """For Simulation Mode: load client profiles/traces

    Args:
        file_path (string): File path for the client profiles/traces

    Returns:
        dictionary: Return the client profiles/traces

    """
    global_client_profile = {}
    if os.path.exists(file_path):
        with open(file_path, "rb") as fin:
            # {client_id: [compute, bandwidth]}
            # compute: Gflops
            # bandwidth: Mbps
            global_client_profile = pickle.load(fin)
    return global_client_profile


class ClientPropertyManager:
    def __init__(self, client_profile_path: str):
        self.client_profiles = load_client_profile(file_path=client_profile_path)

    def get_client_profile(self, client_id):
        """
        Args:
            client_id:
        Returns:
            {"computation": xxx, "communication": xxx}
        """
        client_id = client_id % len(self.client_profiles)
        return  self.client_profiles[client_id]

    def get_random_profile(self):
        return random.choice(self.client_profiles)