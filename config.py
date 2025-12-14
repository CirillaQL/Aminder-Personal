import yaml
import os

class Config:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_ai_config(self):
        return self.config.get("ai", {})

# Example usage:
# config = Config()
# ai_settings = config.get_ai_config()
# print(ai_settings)
