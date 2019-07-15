import os

import numpy
import torch
import tqdm

from .. import preprocess
from .. import training

def pipeline(
    fpaths,
    stat_path,
    speaker_id_config_path,
    speaker_dist_config_path,
    use_embedding
):
    """Iterates torch Tensor vector embedding of audio files in fpaths.

    Parameters :
    fpaths : list of str path to WAV audio file.
    stat_path : str path of summary statistics of the original
        training data.
    speaker_id_config_path : str path to Config json for speaker
        identification.
    speaker_dist_config_path : str path to Config json for speaker
        metric-learning/verification.
    use_embedding : bool indicates if the embedding model is used
        rather than cosine similarity with the outputs of speaker
        identification.

    Output :
    iterator of torch FloatTensor vector embedding of size (d), where d
        is the number of latent features.
    """
    speaker_id_config = training.Config(speaker_id_config_path)
    speaker_dist_config = training.Config(speaker_dist_config_path)

    model = training.speaker_identification.search_model(
        model_id=speaker_id_config.model,
        latent_size=speaker_id_config.latent_size
    )

    if use_embedding:
        # Load the final parameters trained from metric-learning verification.
        model.replace_tail(
            latent_size=speaker_id_config.latent_size,
            embed_size=speaker_dist_config.embed_size
        )
        model.load_state_dict(torch.load(
            speaker_dist_config.modelf
        ))
    else:
        model.load_state_dict(torch.load(
            speaker_id_config.modelf
        ))
    
    miu, std = numpy.load(stat_path)
    miu = miu.reshape(-1, 1)
    std = std.reshape(-1, 1)

    model.eval()

    for fpath in tqdm.tqdm(fpaths, ncols=80, desc="Embedding speakers"):
        yield run(fpath, miu, std, model)

def run(fpath, miu, std, model):
    """Return torch Tensor vector embedding of the audio at fpath.

    Parameters :
    fpath : str path to WAV audio file.
    miu : numpy array of shape (freq, 1), mean frequencies per bin.
    std : numpy array of shape (freq, 1), standard deviations of
        frequencies per bin.
    model : torch.nn.Module pre-trained model that maps normalized
        spectrograms of arbitruary length to speaker vector embeddings.

    Output :
    torch FloatTensor vector embedding of size (d), where d is the number
        of latent features.
    """
    # Step 1: file name -> audio array
    assert os.path.isfile(fpath)
    sample = preprocess.Sample(
        dev=0,
        uid=None,  # doesn't matter
        fpath=fpath
    )
    sample.load()

    # Step 2: audio array -> spectrogram
    sample.transform()
    assert sample.spec is not None

    # Step 3: spectrogram -> frequency-normalized spectrogram
    x = (sample.spec - miu) / std

    # Step 4: normalized spectrogram -> vector embedding
    with torch.no_grad():
        tx = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
        embedding = model(tx.float()).squeeze(0)
        assert len(embedding.shape) == 1

    return embedding.clone()
