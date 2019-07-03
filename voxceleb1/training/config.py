import json
import os

from .. import utils


DOB_HEADER = "_dob"
FILE_PATH = "path"

class Config:

    def __init__(self, param_path, dob_header=DOB_HEADER):
        self.param_path = param_path
        self.dob_header = dob_header
        self.param_dict = self._create_dict()

    @staticmethod
    def write(obj, path_header, dob_header=DOB_HEADER):
        fpath = obj[path_header]
        assert not os.path.isfile(fpath)
        assert dob_header not in obj
        obj[dob_header] = utils.format_time()
        with open(fpath, "w") as f:
            json.dump(obj, f)

    def get_creation_time(self):
        return self.param_dict[self.dob_header]

    def __getattr__(self, key):
        return self.param_dict[key]

    def __str__(self):
        return "Config: <%s>\n\t%s" % (
            self.param_path, self._str_params()
        )

    # === PRIVATE ===

    def _create_dict(self):
        with open(self.param_path, "r") as f:
            return json.load(f)

    def _str_params(self):
        return "\n\t".join(
            " = ".join([key, self.param_dict[key]])
            for key in sorted(self.param_dict.keys())
        )
