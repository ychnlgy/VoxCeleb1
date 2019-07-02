import json, os

class Config:

    def __init__(self, param_path):
        self.param_path = param_path
        with open(param_path, "r") as f:
            self.param_dict = json.load(f)

    def __getattr__(self, key):
        return self.param_dict[key]

    def __str__(self):
        return "<Config: %s>" % self.param_path
