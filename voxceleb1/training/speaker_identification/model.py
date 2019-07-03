import abc

import torch


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
