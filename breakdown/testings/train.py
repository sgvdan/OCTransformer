#!/usr/bin/env python

"""
Builds a convolutional neural network on the fashion mnist data set.
Designed to show wandb integration with pytorch.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F
from fashion_data import Kermany_DataSet

import wandb
import os
from torchvision.models import resnet18, resnet50, resnet101, resnet152, wide_resnet50_2, wide_resnet101_2


class dot_dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


hyperparameter_defaults = dict(
    batch_size=100,
    learning_rate=0.001,
    res_pretrain=False,
    optimizer="sgd",
    mom=0.9,
    weight_decay=0.001,
    architecture='res18',
)

wandb.init(config=hyperparameter_defaults, project="pytorch-cnn-fashion-kermany-val-allres")
config = wandb.config


class Resnet18(torch.nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        # self.resnet = resnet18(pretrained=pretrained, num_classes=num_classes)
        self.resnet = resnet18(pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.reshape(batch_size, channels, height, width)
        x = self.resnet(x)
        x = x.reshape(batch_size, -1)

        return x


class Resnet50(torch.nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.resnet = resnet50(pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.reshape(batch_size, channels, height, width)
        x = self.resnet(x)
        x = x.reshape(batch_size, -1)

        return x


class Resnet101(torch.nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.resnet = resnet101(pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.reshape(batch_size, channels, height, width)
        x = self.resnet(x)
        x = x.reshape(batch_size, -1)

        return x


class Resnet152(torch.nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.resnet = resnet152(pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.reshape(batch_size, channels, height, width)
        x = self.resnet(x)
        x = x.reshape(batch_size, -1)

        return x


class Wide_Resnet50_2(torch.nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.resnet = wide_resnet50_2(pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.reshape(batch_size, channels, height, width)
        x = self.resnet(x)
        x = x.reshape(batch_size, -1)

        return x


class Wide_Resnet101_2(torch.nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.resnet = wide_resnet101_2(pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.reshape(batch_size, channels, height, width)
        x = self.resnet(x)
        x = x.reshape(batch_size, -1)

        return x


def main():
    seed = 25
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"running on {device}")
    torch.manual_seed(hash("by removing stochasticity") % seed)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % seed)

    def_args = dot_dict({
        "train": ["../../../data/kermany/train"],
        "val": ["../../../data/kermany/val"],
        "test": ["../../../data/kermany/test"],
    })

    train_dataset = Kermany_DataSet(def_args.train[0])

    val_dataset = Kermany_DataSet(def_args.val[0])

    label_names = [
        "NORMAL",
        "CNV",
        "DME",
        "DRUSEN",
    ]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False)
    model = None
    if config.architecture == "res18":
        model = Resnet18(4, pretrained=config.res_pretrain)
    elif config.architecture == "res50":
        model = Resnet50(4, pretrained=config.res_pretrain)
    elif config.architecture == "res101":
        model = Resnet101(4, pretrained=config.res_pretrain)
    elif config.architecture == "res152":
        model = Resnet152(4, pretrained=config.res_pretrain)
    elif config.architecture == "wide_resnet50_2":
        model = Wide_Resnet50_2(4, pretrained=config.res_pretrain)
    elif config.architecture == "wide_resnet101_2":
        model = Wide_Resnet101_2(4, pretrained=config.res_pretrain)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    wandb.watch(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = None
    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.mom,
                                    weight_decay=config.weight_decay)
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, momentum=config.mom,
                                        weight_decay=config.weight_decay)

    iter = 0
    for epoch in range(3):
        print(f'epoch: {epoch}')
        for i, (images, labels) in enumerate(train_loader):

            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()

            iter += 1
            if iter % 50 == 0:
                print(f'iter : {iter}')
                print(loss)
                wandb.log({"loss": loss, "epoch": epoch})
            if iter % 500 == 0:
                # Calculate Accuracy
                correct = 0.0
                correct_arr = [0.0] * 10
                total = 0.0
                total_arr = [0.0] * 10
                # Iterate through test dataset
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = Variable(images).to(device)
                        labels = labels.to(device)
                        # Forward pass only to get logits/output
                        outputs = model(images)

                        # Get predictions from the maximum value
                        _, predicted = torch.max(outputs.data, 1)

                        # Total number of labels
                        total += labels.size(0)
                        correct += (predicted == labels).sum()

                        for label in range(4):
                            correct_arr[label] += (((predicted == labels) & (labels == label)).sum())
                            total_arr[label] += (labels == label).sum()

                    accuracy = correct / total

                    metrics = {'accuracy': accuracy}
                    for label in range(4):
                        metrics['Accuracy ' + label_names[label]] = correct_arr[label] / total_arr[label]

                    wandb.log(metrics)

                    # wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                    # y_true = ground_truth, preds = predictions,
                    #                                class_names = class_names)})

                    # Print Loss
                    print('Iteration: {0} Loss: {1:.2f} Accuracy: {2:.2f}'.format(iter, loss, accuracy))
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))


if __name__ == '__main__':
    main()
