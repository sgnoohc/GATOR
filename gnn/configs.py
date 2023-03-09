import json
from types import SimpleNamespace

class GatorConfig(SimpleNamespace):
    """
    Class that imports GNN configs

    TODO: add validation
    """
    @classmethod
    def from_json(cls, config_json):
        with open(config_json, "r") as f:
            return json.load(f, object_hook=lambda d: cls(**d))
