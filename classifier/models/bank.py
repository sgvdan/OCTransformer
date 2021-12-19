import os
import pickle
import torch
from pathlib import Path


class ModelsBank:
    def __init__(self, config):
        self.config = config
        self.bank_path = Path(self.config.bank_path)
        self.bank_fs = self.bank_path / '.bank_fs'

        if not self.config.model_bank_open:
            return

        if self.bank_path.exists():
            with open(self.bank_fs, 'rb') as file:
                self.bank = pickle.load(file)
        else:
            self.bank = {}

    def sync_model(self, model, optimizer, epoch, avg_accuracy):
        if not self.config.model_bank_open:
            return

        # Save Model
        if self.config.backup_last_model:
            path = os.path.join(self.bank_path, model.name, 'epoch-{:06d}.tar'.format(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print("Model {model_name} backup saved at {path}".format(model_name=model.name, path=path))

        if self.config.keep_best_model:
            if model.name not in self.bank or avg_accuracy > self.bank[model.name].accuracy:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(self.bank_path, model.name, 'best.tar'))
                self.bank[model.name].accuracy = avg_accuracy
                print("Best model updated.", "Model: {0}, Avg. Accuracy: {1}".format(model.name, avg_accuracy))

    def load_model_env(self, model, optimizer):
        if not self.config.model_bank_open:
            return

        best_model_path = os.path.join(self.bank_path, model.name + ".pth")
        if not os.path.exists(best_model_path):
            print("Model {model_name} does not exist.".format(model_name=model.name))

        print("Loading best model ({model_name}) state from: {{path}}".format(model_name=model.name, path=best_model_path), flush=True)
        states_dict = torch.load(best_model_path)
        model.load_state_dict(states_dict['model_state_dict'])
        optimizer.load_state_dict(states_dict['optimizer_state_dict'])
