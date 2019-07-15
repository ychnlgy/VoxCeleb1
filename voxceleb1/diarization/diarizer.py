import numpy
import torch
import torch.utils.data

from .cluster import Cluster
from .dataset import Dataset

from .. import preprocess
from .. import training

class Diarizer:

    def __init__(
        self,
        speaker_id_config_path,
        speaker_dist_config_path,
        param_path,
        stat_path,
        batch_size,
        num_workers,
        slice_size,
        step_size,
        threshold
    ):
        self._model = self._load_model(
            speaker_id_config_path,
            speaker_dist_config_path,
            param_path
        )
        self._miu, self._std = self._load_stats(param_path)
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._slice_size = slice_size
        self._step_size = step_size
        self._threshold = threshold

    def process(self, fpath):
        spec = self._extract_spectrogram(fpath)
        embeddings = self._extract_embeddings(spec)
        clusters = self._collect_clusters(embeddings)
        joined_clusters = self._join_clusters(clusters)
        return self._clean(joined_clusters)

    def _extract_embeddings(self, spec):
        "Return the speaker embeddings made by the model on the spectrogram."
        dataloader = self._create_dataloader(spec)
        embeddings = []
        with torch.no_grad():
            for X in dataloader:
                embedding = self._model.embed(X)
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
                focus_cluster.append(emb, dt=self._step_size)
            else:
                focus_cluster = self._create_cluster(i, emb)
                clusters.append(focus_cluster)
        return clusters

    def _join_clusters(self, clusters):
        "Relabel the clusters if they belong to the same person."
        raise NotImplementedError

    def _clean(self, joined_clusters):
        "Return a simple format for interpreting the diarization."
        raise NotImplementedError

    def _create_cluster(self, idx, embedding):
        return Cluster(
            index=idx*self._step_size,
            slice_len=self._slice_len,
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
        speaker_dist_config_path,
        param_path
    ):
        speaker_id_config = training.Config(speaker_id_config_path)
        speaker_dist_config = training.Config(speaker_dist_config_path)

        model = training.speaker_identification.search_model(
            model_id=speaker_id_config.model,
            latent_size=speaker_id_config.latent_size
        )
        model.load_state_dict(torch.load(param_path))
        model.eval()
        return model
        
