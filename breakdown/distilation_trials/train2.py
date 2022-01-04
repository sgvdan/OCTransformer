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
from torch.optim.lr_scheduler import StepLR
import sweeps
import os


class dot_dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


config = dot_dict({
    "architecture": "res18",
    "pretrained_res": False,
    "seed": 42,
    "lr": 0.001,
    "epochs": 2,
    "batch_size": 16,
    'optimizer': "sgd",
    "mom": 0.9,
    'scheduler': "lr",
    "scheduler_gamma": 0.9,
    "scheduler_step_size": 1,
    "vit_patch_size": 16,
    "vit_num_classes": 4,
    "vit_embed_dim": 768,
    "vit_depth": 12,
    "vit_num_heads": 12,
    "vit_mlp_ratio": 4.,
    "vit_drop_rate": 0.2,
    "vit_num_patches": 64,
    "vit_attn_drop_rate": 0.,
    'weight_decay': 0.001

})
wandb.login()
wandb.init(project="my-test-project", entity="guylu", config=config, name=str(config))
config = wandb.config


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"running on {device}")
    wandb.name = str(config)
    random.seed(hash("setting random seeds") % config.seed)
    np.random.seed(hash("improves reproducibility") % config.seed)
    torch.manual_seed(hash("by removing stochasticity") % config.seed)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % config.seed)

    def_args = dot_dict({
        "train": ["../../../data/kermany/train"],
        "val": ["../../../data/kermany/val"],
        "test": ["../../../data/kermany/test"],
    })

    print("getting traning set")
    trainset = kermany_dataset.Kermany_DataSet(def_args.train[0])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                              shuffle=True, num_workers=0)  # drop_last=True
    print("getting validation set")
    valset = kermany_dataset.Kermany_DataSet(def_args.val[0])
    valloader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size,
                                            shuffle=False, num_workers=0)
    print("starting network")

    criterion = nn.CrossEntropyLoss()  # nn.L1Loss

    if config.architecture == "vit":
        model = kermany_net.MyViT(config).to(device)
    elif config.architecture == "res18":
        model = kermany_net.Resnet18(4, pretrained=config.pretrained_res).to(device)
    elif config.architecture == "res50":
        model = kermany_net.Resnet50(4, pretrained=config.pretrained_res).to(device)
    elif config.architecture == "res101":
        model = kermany_net.Resnet101(4, pretrained=config.pretrained_res).to(device)
    elif config.architecture == "res152":
        model = kermany_net.Resnet152(4, pretrained=config.pretrained_res).to(device)
    elif config.architecture == "wide_resnet50_2":
        model = kermany_net.Wide_Resnet50_2(4, pretrained=config.pretrained_res).to(device)
    elif config.architecture == "wide_resnet101_2":
        model = kermany_net.Wide_Resnet101_2(4, pretrained=config.pretrained_res).to(device)

    wandb.watch(model)

    if config.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.mom, weight_decay=config.weight_decay)
    elif config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=config.lr, momentum=config.mom,
                                  weight_decay=config.weight_decay)

    if config.scheduler == "lr":
        scheduler = StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)

    #############################################################################

    for epoch in range(config.epochs):  # loop over the dataset multiple times
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
            if i % 100 == 0:  # print every 2000 mini-batches
                wandb.log({"epoch": epoch, "train loss": loss, "train acc": acc})
                if i % 1000 == 0:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        scheduler.step()

    print('Finished Training')

    correct = 0
    with torch.no_grad():
        running_loss = 0.0
        for i, data in enumerate(tqdm(valloader), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            outputs = torch.argmax(outputs, dim=1)
            acc = (outputs == labels).sum().item() / inputs.shape[0]
            correct += (outputs == labels).sum().item()
            wandb.log({"val loss": loss, "val acc": acc})

            # print statistics
            running_loss += loss.item()

    wandb.log({"Final Accuracy": correct / len(valset)})
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
    print('Finished Validating')
    print()
    print("Finito La Cola")


if __name__ == '__main__':
    # sweep_id = wandb.sweep(sweeps.sweeps_config_resnet)
    # wandb.agent(sweep_id, function=main)
    # print("hi")
    main()

# model.to_onnx()
# wandb.save("model.onnx")
