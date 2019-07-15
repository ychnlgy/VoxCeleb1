import os

import numpy
import pandas
import torch
import tqdm

from .equal_error import equal_error
from .pipeline import pipeline

def main(
    veri_test_path,
    dataroot,
    stat_path,
    speaker_id_config_path,
    speaker_dist_config_path,
    use_embedding
):
    df = _parse_pair_file(veri_test_path)
    file_set = _collect_file_set(df)
    embedding_map = _map_file_to_embedding(
        dataroot, file_set, stat_path,
        speaker_id_config_path,
        speaker_dist_config_path,
        use_embedding
    )
    dist_fn = [cosine_sim, euclidean_dist][use_embedding]
    dist = _compute_all_pair_dist(df, embedding_map, dist_fn)
    eer_scores = equal_error(df.label.values, dist)
    print("EER: %.4f\nThreshold: %f" % eer_scores)

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

def _map_file_to_embedding(
    dataroot, file_set, stat_path,
    speaker_id_config_path,
    speaker_dist_config_path,
    use_embedding
):
    fpaths = [os.path.join(dataroot, f) for f in file_set]
    embeddings = list(pipeline(
        fpaths, stat_path,
        speaker_id_config_path,
        speaker_dist_config_path,
        use_embedding
    ))
    assert len(file_set) == len(embeddings)
    return dict(zip(file_set, embeddings))

def _compute_all_pair_dist(df, embedding_map, dist_fn):
    yh = numpy.zeros(len(df))
    for i, label, f1, f2 in df.itertuples():
        yh[i] = _compute_pair_dist(f1, f2, embedding_map, dist_fn)
    return yh

def euclidean_dist(v1, v2):
    return (v1 - v2).norm().item()

def cosine_sim(v1, v2):
    return (v1 * v2).sum().item()/v1.norm()/v2.norm()

def _compute_pair_dist(f1, f2, embedding_map, dist_fn):
    v1 = embedding_map[f1]
    v2 = embedding_map[f2]
    return dist_fn(v1, v2).numpy()
