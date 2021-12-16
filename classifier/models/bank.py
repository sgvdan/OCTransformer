import os
import pickle
import torch
from pathlib import Path


class ModelsBank:
    def __init__(self, bank_path, config):
        # READ dict from bank_path
        self.bank_path = Path(bank_path)
        self.config = config

        if self.config.model_bank_open and self.bank_path.exists():
            with open(self.bank_path, 'rb') as file:
                self.bank = pickle.load(file)
        else:
            self.bank = {}

    def sync_model(self, model, optimizer, epoch, avg_accuracy):
        # Save Model
        if self.config.backup_last_model:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join('./models/latest-epoch', '{:06d}.tar'.format(epoch)))

        if self.config.keep_best_model:
            if model.name not in self.bank or avg_accuracy > self.bank[model.name].accuracy:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join('./models/', '{}.tar'.format(model.name)))
                self.bank[model.name].accuracy = avg_accuracy

                print("Best model updated.", "Model: {0}, Avg. Accuracy: {1}".format(model.name, avg_accuracy))
