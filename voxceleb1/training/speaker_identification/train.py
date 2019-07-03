import sys
import os

import torch
import tqdm

import voxceleb1

def train(config, producer, original_model, log):
    if os.path.isfile(config.modelf):
        log.write("Model already trained on speaker identification here: %s" % config.modelf)
        original_model.load_state_dict(torch.load(config.modelf))
        return original_model

    if torch.cuda.is_available():
        device = "gpu"
        model = torch.nn.DataParallel(original_model.to(device))
    else:
        device = "cpu"
        model = original_model

    lossf = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=0.9,
        weight_decay=config.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=config.epochs
    )

    avg = voxceleb1.utils.MovingAverage(momentum=0.95)

    for epoch in range(config.epochs):

        data, test = producer.produce()
        dataloader = voxceleb1.utils.tensor_tools.create_loader(
            data, batch_size=config.batch_size, shuffle=True
        )
        testloader = voxceleb1.utils.tensor_tools.create_loader(
            test, batch_size=config.batch_size*2
        )
        
        model.train()

        with tqdm.tqdm(dataloader, ncols=80) as bar:
            for X, Y in dataloader:
                X = X.to(device)
                Y = Y.to(device)
                
                Yh = model(X)
                loss = lossf(Yh, Y)
                optim.zero_grad()
                loss.backward()
                optim.step()

                avg.update(loss.item())
                bar.set_description("Loss %.4f" % avg.peek())

        model.eval()

        with torch.no_grad():

            correct = n = 0.0

            for X, Y in testloader:

                X = X.to(device)
                Y = Y.to(device)

                Yh = model(X)
                _, pred = Yh.max(dim=1)
                correct += (pred == Y).long().sum().item()
                n += len(Y)

            acc = correct / n * 100.0

            log.write("Epoch %d accuracy: %.2f" % (epoch, acc))

    dpath = os.path.dirname(config.modelf)
    if not os.path.isdir(dpath):
        os.makedirs(dpath)

    torch.save(original_model.to("cpu").state_dict(), config.modelf)
    log.write("Saved model to %s" % config.modelf)
    return original_model
