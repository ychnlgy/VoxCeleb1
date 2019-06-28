import sys

import torch
import tqdm

from .DataProducer import DataProducer

import voxceleb1

def train(params, dataset, model, log):
    producer = DataProducer(params.slice_size, dataset)

    model = torch.nn.DataParallel(model)
    lossf = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(
        model.parameters(),
        lr=params.speaker_identification_lr,
        momentum=0.9,
        weight_decay=params.speaker_identification_weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=params.speaker_identification_epochs
    )

    avg = voxceleb1.utils.MovingAverage(momentum=0.95)

    for epoch in range(params.speaker_identification_epochs):

        data, test = producer.produce()
        dataloader = voxceleb1.utils.tensor_tools.create_loader(
            data, batch_size=params.batch_size, shuffle=True
        )
        testloader = voxceleb1.utils.tensor_tools.create_loader(
            test, batch_size=params.batch_size*2
        )
        
        model.train()

        with tqdm.tqdm(dataloader, ncols=80) as bar:
            for X, Y in dataloader:

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

                Yh = model(X)
                _, pred = Yh.max(dim=1)
                correct += (pred == Y).long().sum().item()
                n += len(Y)

            acc = correct / n * 100.0

            log.write("Epoch %d accuracy: %.2f" % (epoch, acc))
