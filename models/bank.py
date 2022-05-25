import os
import pickle
import string

import torch
from pathlib import Path

from torch.nn import Conv2d
from torch.optim.lr_scheduler import ReduceLROnPlateau

import util
from models.deepset import DeepSet
from models.slivernet import SliverNet2
from models.vit import MyViT
import torchvision.models as tmodels

# from pdmodels import pdresnet18


class ModelsBank:
    def __init__(self, config):
        self.config = config
        self.bank_path = Path('.models_bank')
        self.bank_record_path = self.bank_path / '.bank_record'

        if self.bank_record_path.exists():
            with open(self.bank_record_path, 'rb') as file:
                self.bank_record = pickle.load(file)
        else:
            self.bank_record = {}

    def get_environment(self):
        # Choose Backbone
        backbone_name = self.config.backbone.lower()
        if backbone_name == 'resnet18':
            backbone = tmodels.resnet18(pretrained=False).to(self.config.device)
        elif backbone_name == 'imagenet_resnet18':
            backbone = tmodels.resnet18(pretrained=True).to(self.config.device)
        elif backbone_name == 'kermany_resnet18':
            backbone = tmodels.resnet18(pretrained=False, num_classes=4).to(self.config.device)
            kermany_path = '.models_bank/kermany_resnet18/resnet18.tar'
            states_dict = torch.load(kermany_path)
            backbone.load_state_dict(states_dict['model_state_dict'])
        elif backbone_name == 'kermany_ls_resnet18':
            backbone = tmodels.resnet18(pretrained=False, num_classes=4)
            backbone.conv1 = Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            backbone = backbone.to(self.config.device)
        # elif backbone_name == 'pdresnet18':
        #     backbone = pdresnet18(pretrained=False).to(self.config.device)
        else:
            raise NotImplementedError

        # Choose Model
        model_type = self.config.model.lower()
        if model_type == 'vit':
            print('IMPORTANT: Working in undeterminstic manner - so as to support MGU-Net\'s max_pool operation')
            torch.use_deterministic_algorithms(mode=False)
            backbone.fc = torch.nn.Linear(in_features=512, out_features=self.config.embedding_dim,
                                          device=self.config.device)
            model = MyViT(backbone, self.config).to(self.config.device)

        elif model_type == 'slivernet':
            assert self.config.batch_size == 1  # Only supports batch_size of 1
            print('IMPORTANT: Working in undeterminstic manner - so as to support SliverNet\'s max_pool operation')
            torch.use_deterministic_algorithms(mode=False)

            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
            model = SliverNet2(backbone=backbone, n_out=len(self.config.labels))
            if self.config.device == 'cuda':
                model = model.cuda()

        elif model_type == 'deepset':
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
            model = DeepSet(backbone=backbone, x_dim=1024, d_dim=self.config.embedding_dim,
                            num_classes=len(self.config.labels)).to(self.config.device)
        else:
            raise NotImplementedError

        # if Model Name not specified, random out a new one
        model_name = self.config.model_name
        if model_name is None:
            model_name = ''.join(util.get_pseudo_random().choices(string.ascii_letters + string.digits, k=10))
            print('Random model name generated:', model_name)
        model.name = model_name

        # Choose Optimizer
        if self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        else:
            raise NotImplementedError

        # Choose Scheduler
        if self.config.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, 'min')
        elif self.config.scheduler is None:
            scheduler = None
        else:
            raise NotImplementedError

        model.thresholds = None

        if self.config.load_best_model:
            self.load_best(model, optimizer, scheduler)

        return model, optimizer, scheduler

    def get_balanced_criterion(self, train_loader):
        class_weights = util.get_balance_class_weights(train_loader.dataset.get_labels())

        # Choose Criterion
        if self.config.criterion == 'cross_entropy':
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device=self.config.device))
        elif self.config.criterion == 'binary_cross_entropy':
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device=self.config.device))
        else:
            raise NotImplementedError

        return criterion

    def sync_model(self, model, optimizer, scheduler, score):
        if not self.config.keep_best_model:
            return

        # Create required subdirectories
        model_path = os.path.join(self.bank_path, model.name)

        if model.name not in self.bank_record:
            self.bank_record[model.name] = {}
            self.bank_record[model.name]['score'] = 0
            self.bank_record[model.name]['thresholds'] = None
            os.makedirs(model_path, exist_ok=True)

        if score > self.bank_record[model.name]['score']:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': None if scheduler is None else scheduler.state_dict(),
            }, os.path.join(self.bank_path, model.name, 'best.tar'))
            self.bank_record[model.name]['score'] = score
            self.bank_record[model.name]['thresholds'] = model.thresholds
            print("Best model updated.", "Model: {0}, Score (Macro-F1): {1}".format(model.name, score))

        # Save bank records
        with open(self.bank_record_path, 'wb+') as file:
            pickle.dump(self.bank_record, file)

    def load_best(self, model, optimizer, scheduler):
        best_model_path = os.path.join(self.bank_path, model.name, 'best.tar')
        if not os.path.exists(best_model_path):
            print("Model {model_name} does not exist.".format(model_name=model.name))
            return

        print("Loading best model ({0}) state from: {1}".format(model.name, best_model_path), flush=True)
        states_dict = torch.load(best_model_path)
        model.load_state_dict(states_dict['model_state_dict'])
        optimizer.load_state_dict(states_dict['optimizer_state_dict'])
        if scheduler is not None and states_dict['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(states_dict['scheduler_state_dict'])

        model.thresholds = self.bank_record[model.name]['thresholds']
