import os
import pickle
import string
import random

import torch
from pathlib import Path

from models.vit import MyViT


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
        # Choose Model
        if self.config.model == 'vit':
            model = MyViT(self.config).to(self.config.device)
        else:
            raise NotImplementedError

        # if Model Name not specified, random out a new one
        if self.config.model_name is None:
            model_name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            print('Random model name generated:', model_name)
        model.name = model_name

        # Choose Optimizer
        if self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        else:
            raise NotImplementedError

        # Choose Criterion
        if self.config.criterion == 'cross_entropy':
            criterion = torch.nn.functional.cross_entropy
        else:
            raise NotImplementedError

        if self.config.load_best_model:
            self.load_best(model, optimizer)

        return model, criterion, optimizer

    def sync_model(self, model, optimizer, avg_accuracy):
        if not self.config.keep_best_model:
            return

        # Create required subdirectories
        model_path = os.path.join(self.bank_path, model.name)

        if model.name not in self.bank_record:
            self.bank_record[model.name] = {}
            self.bank_record[model.name]['accuracy'] = 0
            os.makedirs(model_path, exist_ok=True)

        if avg_accuracy > self.bank_record[model.name]['accuracy']:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(self.bank_path, model.name, 'best.tar'))
            self.bank_record[model.name]['accuracy'] = avg_accuracy
            print("Best model updated.", "Model: {0}, Avg. Accuracy: {1}".format(model.name, avg_accuracy))

        # Save bank records
        with open(self.bank_record_path, 'wb+') as file:
            pickle.dump(self.bank_record, file)

    def load_best(self, model, optimizer):
        best_model_path = os.path.join(self.bank_path, model.name, 'best.tar')
        if not os.path.exists(best_model_path):
            print("Model {model_name} does not exist.".format(model_name=model.name))
            return

        print("Loading best model ({0}) state from: {1}".format(model.name, best_model_path), flush=True)
        states_dict = torch.load(best_model_path)
        model.load_state_dict(states_dict['model_state_dict'])
        optimizer.load_state_dict(states_dict['optimizer_state_dict'])
