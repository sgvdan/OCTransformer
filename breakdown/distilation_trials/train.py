import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import kermany_dataset
import kermany_net
from config import default_config
import utils
import wandb
import random
import numpy as np
import argparse
import cv2 as cv

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"running on {device}")
wandb.login()
wandb.init(project="my-test-project", entity="guylu")
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2 ** 32 - 1)
np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build Kermany dataset')

    parser.add_argument('--train', type=str, nargs='+', default=None)
    parser.add_argument('--val', type=str, nargs='+', default=None)
    parser.add_argument('--test', type=str, nargs='+', default=None)

    args = parser.parse_args()

    wandb.config = {
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 128
    }

    batch_size = 10
    epochs = 50
    print("getting traning set")
    trainset = kermany_dataset.Kermany_DataSet(args.train[0])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    print("getting validation set")
    valset = kermany_dataset.Kermany_DataSet(args.val[0])
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)
    print("preping network")
    config = 0
    model = kermany_net.Resnet(4).to(device)
    wandb.watch(model)
    criterion = nn.CrossEntropyLoss()  # nn.L1Loss
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            acc = (outputs == labels).sum().item() / inputs.shape[0]
            wandb.log({"epoch": epoch, "train loss": loss, "acc": acc})
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    with torch.no_grad():
        running_loss = 0.0
        for i, data in enumerate(valloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            acc = (outputs == labels).sum().item() / inputs.shape[0]
            wandb.log({"val loss": loss, "acc": acc})
            optimizer.step()

            # print statistics
            running_loss += loss.item()

    print('Finished Validating')
    # model.to_onnx()
    # wandb.save("model.onnx")
