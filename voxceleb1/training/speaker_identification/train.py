import sys
import os

import torch
import tqdm

import torch.utils.data

import voxceleb1

def train(config, dataset, testset, cores, original_model, log):

    log.write("Part 1: speaker identification")
    
    if os.path.isfile(config.modelf):
        log.write(
            "Model already trained " \
            "on speaker identification here: " \
            "%s" % config.modelf
        )
        original_model.load_state_dict(torch.load(config.modelf))
        return original_model

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=cores
    )

    params = voxceleb1.utils.tensor_tools.param_count(original_model)
    log.write("Model parameters: %d" % params)

    if torch.cuda.is_available():
        device = "cuda"
        model = torch.nn.DataParallel(original_model.to(device))
    else:
        device = "cpu"
        model = original_model

    log.write("Using device: %s" % device)

    lossf = torch.nn.CrossEntropyLoss()
    log.write("Loss function: %s" % lossf)
    
    optim = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=0.9,
        weight_decay=config.weight_decay
    )
    log.write("Optimizer:\n%s" % optim)
    
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=config.epochs
    )
    log.write("Learning rate scheduler:\n%s" % sched)

    avg = voxceleb1.utils.MovingAverage(momentum=0.95)

    log.write("Began training...")

    for epoch in range(config.epochs):

        scorer = voxceleb1.utils.TopClassPredictor(top_ks=[1, 5])
        
        model.train()

        with tqdm.tqdm(dataloader, ncols=80) as bar:
            for X, Y in bar:
                X = X.to(device)
                Y = Y.to(device)
                
                Yh = model(X)
                loss = lossf(Yh, Y)
                optim.zero_grad()
                loss.backward()
                optim.step()

                avg.update(loss.item())
                bar.set_description("Loss %.4f" % avg.peek())

                scorer.update(Yh, Y)

        top1, top5 = scorer.peek()
        top1 *= 100.0
        top5 *= 100.0
        log.write(
            "Epoch %d/%d training top-1/top-5 accuracy: " \
            "%.2f%%/%.2f%%" % (epoch+1, config.epochs, top1, top5)
        )
        
    dpath = os.path.dirname(config.modelf)
    if not os.path.isdir(dpath):
        os.makedirs(dpath)

    torch.save(original_model.cpu().state_dict(), config.modelf)
    log.write("Saved model to %s" % config.modelf)
    return original_model
