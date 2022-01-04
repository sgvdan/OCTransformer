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
from torchvision.models import resnet18

class dot_dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


hyperparameter_defaults = dict(
    batch_size=100,
    learning_rate=0.001,
    epochs=2,
)

wandb.init(config=hyperparameter_defaults, project="pytorch-cnn-fashion")
config = wandb.config


class Resnet18(torch.nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.resnet = resnet18(pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.reshape(batch_size, channels, height, width)
        x = self.resnet(x)
        x = x.reshape(batch_size, -1)

        return x


def main():

    def_args = dot_dict({
        "train": ["../../../data/kermany/train"],
        "val": ["../../../data/kermany/val"],
        "test": ["../../../data/kermany/test"],
    })

    train_dataset = Kermany_DataSet(def_args.train[0])

    test_dataset = Kermany_DataSet(def_args.val[0])

    label_names = [
        "T-shirt or top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Boot"]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False)

    model = Resnet18(4)
    wandb.watch(model)
    print("got data")
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    iter = 0
    for epoch in range(config.epochs):
        print(epoch)
        for i, (images, labels) in enumerate(train_loader):

            images = Variable(images)
            labels = Variable(labels)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()
            print(loss)
            # Updating parameters
            optimizer.step()

            iter += 1

            if iter % 10 == 0:
                # Calculate Accuracy
                correct = 0.0
                correct_arr = [0.0] * 10
                total = 0.0
                total_arr = [0.0] * 10
                print("total")
                # Iterate through test dataset
                for images, labels in test_loader:
                    images = Variable(images)

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

                metrics = {'accuracy': accuracy, 'loss': loss}
                for label in range(4):
                    metrics['Accuracy ' + label_names[label]] = correct_arr[label] / total_arr[label]

                wandb.log(metrics)

                # Print Loss
                print('Iteration: {0} Loss: {1:.2f} Accuracy: {2:.2f}'.format(iter, loss, accuracy))
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))


if __name__ == '__main__':
    main()
