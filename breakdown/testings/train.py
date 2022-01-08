#!/usr/bin/env python

"""
Builds a convolutional neural network on the fashion mnist data set.
Designed to show wandb integration with pytorch.
"""

from data import Kermany_DataSet
import timm
import wandb
import os

from utils import *
from res_models import *
from model_running import *

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

wandb.init(config=hyperparameter_defaults, project="pytorch-cnn-fashion-kermany_val_test_new_wights_vit")
config = wandb.config


def init():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"running on {device}")
    torch.manual_seed(hash("by removing stochasticity") % wandb.config.seed)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % wandb.config.seed)
    def_args = dot_dict({
        "train": ["../../../data/kermany/train"],
        "val": ["../../../data/kermany/val"],
        "test": ["../../../data/kermany/test"],
    })
    label_names = [
        "NORMAL",
        "CNV",
        "DME",
        "DRUSEN",
    ]
    return def_args, device, label_names


def Get_Model(config, device):
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
    return model


def Handle_Data(def_args):
    train_dataset = Kermany_DataSet(def_args.train[0])
    val_dataset = Kermany_DataSet(def_args.val[0])
    test_dataset = Kermany_DataSet(def_args.test[0])

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
    return test_loader, train_loader, val_loader


def Get_Optimizer(model):
    optimizer = None
    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.mom,
                                    weight_decay=config.weight_decay)
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, momentum=config.mom,
                                        weight_decay=config.weight_decay)
    return optimizer


def main():
    def_args, device, label_names = init()

    print("gettin data")
    test_loader, train_loader, val_loader = Handle_Data(def_args)

    print("gettin model")
    model = Get_Model(config, device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Get_Optimizer(model)
    #########################################################################################################
    #                                                 TRAINING
    #########################################################################################################

    print("starting training:\n\n")
    Train(criterion, device, label_names, model, optimizer, train_loader, val_loader)

    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))

    #########################################################################################################
    #                                                 TESTING
    #########################################################################################################
    print("TESTING TIMZZZ")

    Testing(device, label_names, model, test_loader)

    print("finito la musica")


if __name__ == '__main__':
    main()
    print("\n\n\n\nWe done did it mates")
