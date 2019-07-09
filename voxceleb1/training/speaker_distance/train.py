import os

import torch

from .loss import triplet_loss

import voxceleb1


def train(config, producer, model, log):
    
    if os.path.isfile(config.modelf):
        log.write(
            "Model already trained " \
            "on metric learning here: " \
            "%s" % config.modelf
        )
        model.load_state_dict(torch.load(config.modelf))
        return model

    if torch.cuda.is_available():
        device = "cuda"
        model = torch.nn.DataParallel(model.to(device))
    else:
        device = "cpu"

    lossf = triplet_loss.batch_hard
    log.write("Loss function: triplet loss (batch-hard)")
    
    optim = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=0.9,
        weight_decay=config.weight_decay
    )
    log.write("Optimizer:\n%s" % optim)
    
    avg = voxceleb1.utils.MovingAverage(momentum=0.95)

    log.write("Began training...")
