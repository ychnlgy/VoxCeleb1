# Think about dense nets
# Todo : write tail

import abc

import torch

import voxceleb1


def search_model(model_id, features, latent_size, unique_labels):
    return _get_model_dict()[model_id](features, latent_size, unique_labels)

def _get_model_dict():
    return {Model.ID: Model for Model in _BaseModel.__subclasses__()}

class _BaseModel(abc.ABC, torch.nn.Module):

    def __init__(self, features, latent_size, unique_labels):
        torch.nn.Module.__init__(self)
        self._main = torch.nn.Sequential(
            *self.make_main_layers(features, latent_size)
        )
        self._tail = torch.nn.Sequential(
            *self.make_tail_layers(latent_size, unique_labels)
        )

    def extract(self, X):
        return self._main(X)

    def forward(self, X):
        return self._tail(self.extract(X))

    @abc.abstractmethod
    def make_main_layers(self, features, latent_size):
        "Return the list of modules for extracting latent features."

    @abc.abstractmethod
    def make_tail_layers(self, latent_sie, unique_labels):
        "Return the list of modules for transforming latent -> predict."

class _OneClassModule(torch.nn.Module):

    def __init__(self, unique_labels):
        super().__init__()
        self.params = torch.nn.Parameter(
            torch.rand(1, unique_labels)
        )

    def forward(self, X):
        return self.params.repeat(len(X), 1)

class _OneClass(_BaseModel):

    ID = "one-class"

    def make_main_layers(self, features, latent_size):
        "Return the list of modules for extracting latent features."
        return []

    def make_tail_layers(self, latent_size, unique_labels):
        "Return the list of modules for transforming latent -> predict."
        return [_OneClassModule(unique_labels)]

class _ResModel:

    def create_sweep(self, k, w):
        return torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(w, w, (k, 3), padding=(0, 1), bias=False),
            torch.nn.BatchNorm2d(w)
        )

    def create_pipe(self, w):
        return torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(w, w, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(w)
        )

    def create_res_pipe(self, w):
        return voxceleb1.neuron.ResBlock(
            block=voxceleb1.neuron.Parallel(
                br1=self.create_pipe_branch(w),
                br2=self.create_pipe_branch(w)
            ),
        )

    def create_funnel(self, w):
        return torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(w, w*2, 3, padding=1, bias=False, stride=2),
            torch.nn.BatchNorm2d(w*2)
        )

    def create_res_funnel(self, w):
        return voxceleb1.neuron.ResBlock(
            block=voxceleb1.neuron.Parallel(
                br1=self.create_funnel_branch(w),
                br2=self.create_funnel_branch(w)
            ),
            shortcut=voxceleb1.neuron.Shortcut(w, w*2, stride=2)
        )

    def create_funnel_branch(self, w):
        return torch.nn.Sequential(
            self.create_funnel(w),
            self.create_pipe(w*2),
        )

    def create_pipe_branch(self, w):
        return torch.nn.Sequential(
            self.create_pipe(w),
            self.create_pipe(w)
        )

class _ShortRes(_BaseModel, _ResModel):
    "Assume the input is of (N, 1, 256, >300)."

    ID = "short-res"

    def make_main_layers(self, features, latent_size):
        "Return the list of modules for extracting latent features."
        return [
            # (256, 300) -> (128, 150)
            torch.nn.Conv2d(1, 64, 7, padding=3, bias=False, stride=2),

            self.create_res_pipe(64),
            self.create_res_pipe(64),

            # (128, 150) -> (64, 75)
            self.create_res_funnel(64), 
            self.create_res_pipe(128),

            # (64, 75) -> (32, 38)
            self.create_res_funnel(128),
            self.create_res_pipe(256),

            # (32, 38) -> (16, 19)
            self.create_res_funnel(256),
            self.create_res_pipe(512),

            # (16, 19) -> (1, 9)
            voxceleb1.neuron.Parallel(
                br1=self.create_sweep(16, latent_size),
                br2=self.create_sweep(16, latent_size),
            ),

            torch.nn.ReLU(),
            # Global average pool along time: (1, 9) -> ()
            voxceleb1.neuron.Operation(lambda X: X.mean(dim=-1).squeeze(-1)),
        ]

    def make_tail_layers(self, latent_size, unique_labels):
        "Return the list of modules for transforming latent -> predict."
        return [
            torch.nn.Linear(latent_size, unique_labels)
        ]
