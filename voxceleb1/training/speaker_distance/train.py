import os

import torch

from .loss import triplet_loss

import voxceleb1


def train(config, dataset, testset, cores, model, log):

    log.write("Part 2: metric learning")
    
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

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=cores
    )
    
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config.batch_size*2,
        num_workers=cores
    )

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

    for epoch in range(config.epochs):

        model.train()

        with tqdm.tqdm(dataloader, ncols=80) as bar:
            for X in bar:
                X = X.to(device)
                features = model.extract(X)
                loss = lossf(features)

                optim.zero_grad()
                loss.backward()
                optim.step()

                avg.update(loss.item())

        model.eval()

        with torch.no_grad():
            
