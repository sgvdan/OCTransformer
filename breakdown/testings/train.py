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
from timm.models.vision_transformer import VisionTransformer
import timm
import wandb
import os
from torchvision.models import resnet18, resnet50, resnet101, resnet152, wide_resnet50_2, wide_resnet101_2


class dot_dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


hyperparameter_defaults = dict(
    seed=25,
    batch_size=2,
    learning_rate=1e-4,
    res_pretrain=False,
    optimizer="adam",
    mom=0.9,
    weight_decay=0,
    architecture='vit',
    vit_architecture='vit_base_patch16_224',
    vit_pretrain=False,
)

wandb.init(config=hyperparameter_defaults, project="pytorch-cnn-fashion-kermany-val-vit_my_new_dan")
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


def make_weights_for_balanced_classes(dataset, classes):
    num_classes = len(classes)
    num_scans = len(dataset)
    labels = dataset.get_labels()

    # Count # of appearances per each class
    count = [0] * num_classes
    for label in labels:
        count[int(label)] += 1

    # Each class receives weight in reverse proportion to its # of appearances
    weight_per_class = [0.] * num_classes
    for idx in classes:
        weight_per_class[idx] = float(num_scans) / float(count[idx])

    # Assign class-corresponding weight for each element
    weights = [0] * num_scans
    for idx, label in enumerate(labels):
        weights[idx] = weight_per_class[int(label)]

    return torch.FloatTensor(weights)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"running on {device}")
    torch.manual_seed(hash("by removing stochasticity") % wandb.config.seed)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % wandb.config.seed)

    def_args = dot_dict({
        "train": ["../../../data/kermany/train"],
        "val": ["../../../data/kermany/val"],
        "test": ["../../../data/kermany/test"],
    })

    train_dataset = Kermany_DataSet(def_args.train[0])
    val_dataset = Kermany_DataSet(def_args.val[0])
    test_dataset = Kermany_DataSet(def_args.test[0])

    label_names = [
        "NORMAL",
        "CNV",
        "DME",
        "DRUSEN",
    ]
    print("gettin data")
    train_weights = make_weights_for_balanced_classes(train_dataset, [i for i in range(4)])
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               # shuffle=True,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False)
    print("gettin model")
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

    if config.architecture == 'vit':
        model = timm.create_model(config.vit_architecture, pretrained=config.vit_pretrain, num_classes=4,
                                  img_size=(496, 512))

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

    print("starting training:\n\n")
    iter = 0
    for epoch in range(3):
        print(f'epoch: {epoch}')
        for i, (images, labels) in enumerate(train_loader):
            if iter == 501:
                break
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

    #########################################################################################################
    #                                                 TESTING
    #########################################################################################################

    correct = 0.0
    correct_arr = [0.0] * 10
    total = 0.0
    total_arr = [0.0] * 10
    predictions = None
    ground_truth = None
    # Iterate through test dataset
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
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

            if i == 0:
                predictions = predicted
                ground_truth = labels
            else:
                predictions = torch.cat((predictions, predicted), 0)
                ground_truth = torch.cat((ground_truth, labels), 0)
        accuracy = correct / total

        metrics = {'Test Accuracy': accuracy}
        for label in range(4):
            metrics['Test Accuracy ' + label_names[label]] = correct_arr[label] / total_arr[label]
        wandb.log(metrics)
        wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                           y_true=ground_truth, preds=predictions,
                                                           class_names=label_names)})


if __name__ == '__main__':
    main()
