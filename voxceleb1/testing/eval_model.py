import pandas

def _parse_pair_file(fpath):
    "Return DataFrame instance of the VoxCeleb1 verification test file."
    return pandas.read_csv(
        fpath,
        header=None,
        names=["label", "f1", "f2"],
        sep=" "
    )

def _collect_file_set(df):
    f1 = set(df.f1.tolist())
    f2 = set(df.f2.tolist())
    return list(f1 | f2)

def _map_file_to_embedding(file_set, model):
    pass
