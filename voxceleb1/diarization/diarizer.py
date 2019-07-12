from .. import training

class Diarizer:

    def __init__(
        self,
        speaker_id_config_path,
        speaker_dist_config_path,
        param_path,
        slice_size,
        step_size
    ):
        self._model = self._load_model(
            speaker_id_config_path,
            speaker_dist_config_path,
            param_path
        )

    def _load_model(
        self,
        speaker_id_config_path,
        speaker_dist_config_path,
        param_path
    ):
        
        
