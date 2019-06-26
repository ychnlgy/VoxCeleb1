import fnmatch
import os
import tqdm

from .Sample import Sample


EXT = ".wav"
ID_PREFIX = "id"


def collect(root):
    return list(_collect(root))


# === PRIVATE ===


def _collect(root):
    """Yield structs describing the speaker and file.

    Parameters :
    root : str path of root data directory.

    ex. <root>/test/id10270/5r0dWxy17C8/00001.wav should produce
    {
        use="test",
        uid=10270,
        file="test/id10270/5r0dWxy17C8/00001.wav",
        data=None
    }

    ex.  <root>/dev/id10646/0TBBsmYsLO0/00001.wav should produce
    {
        use="dev",
        uid=10646,
        file="dev/id10646/0TBBsmYsLO0/00001.wav",
        data=None
    }
    """
    for train_type in os.listdir(root):
        train_path = os.path.join(root, train_type)
        subjects = os.listdir(train_path)
        for uid_str in tqdm.tqdm(subjects, desc="Collecting %s" % train_type):
            assert uid_str.startswith(ID_PREFIX)
            uid = int(uid_str.lstrip(ID_PREFIX))
            subject_dir = os.path.join(train_path, uid_str)
            assert os.path.isdir(subject_dir)
            for fpath in _depth_first_walk(subject_dir):
                yield Sample(train_type, uid, fpath)


def _depth_first_walk(node):
    if os.path.isfile(node):
        if node.endswith(EXT):
            yield node
    else:
        assert os.path.isdir(node)
        for fname in os.listdir(node):
            dname = os.path.join(node, fname)
            for fpath in _depth_first_walk(dname):
                yield fpath
