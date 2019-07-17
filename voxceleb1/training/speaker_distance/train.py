import os

import torch
import tqdm

from .loss import triplet_loss

import voxceleb1


def train(config, dataset, testset, cores, original_model, log):

    log.write("Part 2: metric learning")
    
    if os.path.isfile(config.modelf):
        log.write(
            "Model already trained " \
            "on metric learning here: " \
            "%s" % config.modelf
        )
        original_model.load_state_dict(torch.load(config.modelf))
        return original_model

    if config.continue_training:
        log.write(
            "Continue training parameters from:\n%s" \
            % config.continue_training_modelf
        )
        original_model.load_state_dict(torch.load(
            config.continue_training_modelf
        ))

    original_model.set_extract_optimize_flag(config.optimize_tail_only)

    if torch.cuda.is_available():
        device = "cuda"
        original_model = original_model.to(device)
        model = torch.nn.DataParallel(original_model)
    else:
        device = "cpu"
        model = original_model

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

    params = list(original_model.get_optimizing_params())
    
    optim = torch.optim.SGD(
        params,
        lr=config.lr,
        momentum=0.9,
        weight_decay=config.weight_decay
    )

    log.write("Optimizer:\n%s" % optim)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=config.epochs
    )
    log.write("Learning rate scheduler:\n%s" % sched)

    param_count = sum(
        torch.numel(p)
        for p in params
        if p.requires_grad
    )
    log.write("Parameters to optimize: %d" % param_count)
    
    avg = voxceleb1.utils.MovingAverage(momentum=0.95)

    log.write("Began training...")

    for epoch in range(config.epochs):

        model.train()

        with tqdm.tqdm(dataloader, ncols=80) as bar:
            for X in bar:
                X = X.to(device)
                features = model(X)
                loss = lossf(features)

                optim.zero_grad()
                loss.backward()
                optim.step()

                avg.update(loss.item())
                bar.set_description("Loss: %.4f" % avg.peek())

            sched.step()

        log.write("Epoch %d/%d loss: %.4f" % (epoch, config.epochs, avg.peek()))

    dname = os.path.dirname(config.modelf)
    if not os.path.isdir(dname):
        os.makedirs(dname)
        
    original_model = original_model.cpu()
    torch.save(original_model.state_dict(), config.modelf)
    log.write("Saved model to %s" % config.modelf)
    
    return model
