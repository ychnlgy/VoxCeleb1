import os

import numpy
import torch
import torch.utils.data
import tqdm

import sklearn.cluster
import sklearn.metrics.pairwise

from .cluster import Cluster
from .cluster_cossim import ClusterCossim
from .dataset import Dataset

from .. import preprocess
from .. import training

class Diarizer:

    def __init__(
        self,
        root,
        speaker_id_config_path,
        speaker_dist_config_path,
        stat_path,
        batch_size,
        num_workers,
        slice_size,
        step_size,
        threshold,
        use_embedding,
        min_samples
    ):
        self._root = root
        self._device = ["cpu", "cuda"][torch.cuda.is_available()]
        self._use_embedding = use_embedding
        self._model = self._load_model(
            speaker_id_config_path,
            speaker_dist_config_path
        )
        self._miu, self._std = self._load_stats(stat_path)
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._slice_size = slice_size
        self._step_size = step_size
        self._threshold = threshold
        self._min_samples = min_samples

    def process(self, fpath):
        spec = self._extract_spectrogram(fpath)
        embeddings = self._extract_embeddings(spec, fpath)

        return self._dbscan(embeddings)
        #clusters = self._collect_clusters(embeddings)
        #joined_clusters = self._join_clusters(clusters)
        #return self._clean(joined_clusters)

    def _dbscan(self, embeddings):
        metric = ["cosine", "euclidean"][self._use_embedding]
        dbscan = sklearn.cluster.DBSCAN(
            eps=self._t,
            min_samples=self._min_samples,
            metric=metric
        )
        dbscan.fit(embeddings.numpy())
        return dbscan.labels_

    def _extract_embeddings(self, spec, fpath):
        "Return the speaker embeddings made by the model on the spectrogram."
        dataloader = self._create_dataloader(spec)
        embeddings = []
        fname = os.path.basename(fpath)
        with torch.no_grad():
            for X in tqdm.tqdm(dataloader, ncols=80, desc="Processing %s" % fname):
                embedding = self._model(X.to(self._device))
                embeddings.append(embedding)
        return torch.cat(embeddings, dim=0)

    def _collect_clusters(self, embeddings):
        "Group the embeddings locally."
        assert len(embeddings.shape) == 2
        focus_cluster = self._create_cluster(0, embeddings[0])
        clusters = [focus_cluster]
        for i in range(1, embeddings.size(0)):
            emb = embeddings[i]
            if focus_cluster.matches(emb):
                focus_cluster.append(emb, dt=1)
            else:
                focus_cluster = self._create_cluster(i, emb)
                clusters.append(focus_cluster)
        return clusters

    def _join_clusters(self, clusters):
        "Relabel the clusters if they belong to the same person."
        num_clusters = len(clusters)
        if num_clusters == 0:
            return clusters
        else:
            clusters[0].set_label(0)
            if num_clusters > 1:
                c1 = clusters[1]
                c1.set_label(1)
                clusters = self._cluster_clusters(
                    labelled_clusters=clusters[:2],
                    todo=clusters[2:]
                )
        return clusters

    def _cluster_clusters(self, labelled_clusters, todo):
        cluster_id = 2
        for cluster in todo:
            matched = False
            for labelled_cluster in reversed(labelled_clusters):
                if labelled_cluster.matches(cluster.average()):
                    cluster.copy_label(labelled_cluster)
                    matched = True
                    break
            if not matched:
                assert cluster.get_label() is None
                cluster.set_label(cluster_id)
                cluster_id += 1

            # All clusters are retained
            labelled_clusters.append(cluster)
        return labelled_clusters

    def _clean(self, joined_clusters):
        "Return a numpy array of labels as representation of diarization."
        out = numpy.zeros(joined_clusters[-1].get_slice().stop)
        for cluster in joined_clusters:
            out[cluster.get_slice()] = cluster.get_label()
        return out

    def _create_cluster(self, idx, embedding):
        ClusterClass = [ClusterCossim, Cluster][self._use_embedding]
        return ClusterClass(
            index=idx,
            slice_len=1,
            embedding=embedding,
            threshold=self._threshold
        )

    def _create_dataloader(self, spec):
        dataset = Dataset(spec, self._slice_size, self._step_size)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers
        )

    def _extract_spectrogram(self, fpath):
        sample = preprocess.Sample(dev=0, uid=None, fpath=fpath)
        sample.load()
        sample.transform()
        spec = (sample.spec - self._miu) / self._std
        return torch.from_numpy(spec).float()

    def _load_stats(self, param_path):
        miu, std = numpy.load(param_path)
        miu = miu.reshape(-1, 1)
        std = std.reshape(-1, 1)
        return miu, std

    def _load_model(
        self,
        speaker_id_config_path,
        speaker_dist_config_path
    ):
        speaker_id_config = training.Config(speaker_id_config_path)

        if self._use_embedding:
            speaker_dist_config = training.Config(speaker_dist_config_path)
            param_path = speaker_dist_config.modelf
            unique_labels = None
        else:
            param_path = speaker_id_config.modelf
            unique_labels = 1251

        model = training.speaker_identification.search_model(
            model_id=speaker_id_config.model,
            latent_size=speaker_id_config.latent_size,
            unique_labels=unique_labels
        )
        
        model.load_state_dict(torch.load(
            os.path.join(self._root, param_path)
        ))
        model.eval()
        return model.to(self._device)
