import json, os

class Config:

    def __init__(self, param_path):
        self.param_path = param_path
        with open(param_path, "r") as f:
            self.param_dict = json.load(f)

    def __getattr__(self, key):
        return self.param_dict[key]

    def __str__(self):
        return "<config: %s>\n\t%s\n</config>" % (
            self.param_path, self.str_params()
        )

    def str_params(self):
        return "\n\t".join(
            " = ".join(item)
            for item in sorted(self.param_dict.items())
        )
