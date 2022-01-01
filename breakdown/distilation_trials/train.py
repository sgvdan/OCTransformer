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
from tqdm import tqdm

class dot_dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

wandb.config = dot_dict({
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 256
})

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"running on {device}")
wandb.login()
wandb.init(project="my-test-project", entity="guylu",name="vit trial")
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

    print("getting traning set")
    trainset = kermany_dataset.Kermany_DataSet(args.train[0])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=wandb.config.batch_size,
                                              shuffle=True, num_workers=0)
    print("getting validation set")
    valset = kermany_dataset.Kermany_DataSet(args.val[0])
    valloader = torch.utils.data.DataLoader(valset, batch_size=wandb.config.batch_size,
                                            shuffle=False, num_workers=0)
    print("starting network")
    config = 0
    model = kermany_net.MyViT() #kermany_net.Resnet(4).to(device)
    wandb.watch(model)
    criterion = nn.CrossEntropyLoss()  # nn.L1Loss
    optimizer = optim.SGD(model.parameters(), lr=wandb.config.lr, momentum=0.9)
    for epoch in range(wandb.config.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        print(f"epoch:{epoch}")
        for i, data in enumerate(tqdm(trainloader), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            outputs = torch.argmax(outputs, dim=1)
            acc = (outputs == labels).sum().item() / inputs.shape[0]
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 0:  # print every 2000 mini-batches
                wandb.log({"epoch": epoch, "train loss": loss, "train acc": acc})
                if i % 100 == 0:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    with torch.no_grad():
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = (outputs == labels).sum().item() / inputs.shape[0]
            wandb.log({"val loss": loss, "val acc": acc})

            # print statistics
            running_loss += loss.item()

    print('Finished Validating')
    print("Finito La Cola")
    # model.to_onnx()
    # wandb.save("model.onnx")
