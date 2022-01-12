import pickle
from functools import partial

import torch
import wandb

import util
from data.hadassah.hadassah_builder2 import *
from data.hadassah.hadassah_data2 import HadassahDataset
from data.kermany.kermany_data import KermanyDataset, get_kermany_transform
from models.bank import ModelsBank
from config import default_config
from data.cache import Cache
from data.data import get_balance_weights
from data.hadassah.hadassah_data import E2EVolumeDataset, get_hadassah_transform
from logger import Logger
from train.train import Trainer


class Experiment:
    def __init__(self, config):
        assert config is not None
        self.config = config

        util.make_deterministic()

        # Initiate W&B
        if self.config.log:
            wandb.login()
            wandb.init(project=self.config.project, group=self.config.log_group, config=self.config)
            self.config = wandb.config

        # Set up Logger
        self.logger = Logger(self.config)

        # Set up Data
        self.train_loader, self.eval_loader, self.test_loader = self.setup_data2()

        # Set up Model Bank
        self.model_bank = ModelsBank(self.config)

        # Set up Trainer
        self.trainer = Trainer(self.config, self.train_loader, self.eval_loader,
                               self.test_loader, self.model_bank, self.logger)

        # Set up Model Environment
        self.model, self.criterion, self.optimizer = self.model_bank.get_environment()

    def setup_data2(self):
        with open(self.config.records_path, 'rb') as file:
            records = pickle.load(file)

        control, study = records.slice_patients('DME', 'DME-END', 1, False, False)

        train_size = 0.65
        eval_size = 0.1
        test_size = 0.25
        train_control, eval_control, test_control = util.split_list(control, [train_size, eval_size, test_size])
        train_study, eval_study, test_study = util.split_list(study, [train_size, eval_size, test_size])

        # Test Loader
        test_dataset = HadassahDataset([*test_control, *test_study], label_name='DME',
                                       transformations=get_hadassah_transform(self.config.input_size))
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.config.batch_size)

        # Evaluation Loader
        eval_dataset = HadassahDataset([*eval_control, *eval_study], label_name='DME',
                                       transformations=get_hadassah_transform(self.config.input_size))
        eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=self.config.batch_size)

        # Train Loader
        train_dataset = HadassahDataset([*train_control, *train_study], label_name='DME',
                                        transformations=get_hadassah_transform(self.config.input_size))
        train_weights = get_balance_weights(train_dataset, num_classes=self.config.num_classes)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.config.batch_size,
                                                   sampler=train_sampler)

        return train_loader, eval_loader, test_loader

    def setup_data(self):
        if self.config.dataset == 'hadassah':
            dataset = partial(E2EVolumeDataset, samples_num=self.config.num_slices)
            transform = get_hadassah_transform(self.config.input_size)
        elif self.config.dataset == 'kermany':
            dataset = KermanyDataset
            transform = get_kermany_transform(self.config.input_size)

        test_dataset = dataset(Cache(self.config.test_cache), transformations=transform)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.config.batch_size)

        eval_dataset = dataset(Cache(self.config.eval_cache), transformations=transform)
        eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=self.config.batch_size)

        train_dataset = dataset(Cache(self.config.train_cache), transformations=transform)
        train_weights = get_balance_weights(train_dataset)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.config.batch_size,
                                                   sampler=train_sampler)

        return train_loader, eval_loader, test_loader

    def run(self):
        self.trainer.train(self.model, self.criterion, self.optimizer, self.config.epochs)

        if self.config.load_best_model:
            self.model_bank.load_best(self.model, self.optimizer)  # Refresh model (important for train over fitting)

        accuracy = self.trainer.test(self.model)
        self.logger.log({'Overall_Accuracy': accuracy})


def main():
    experiment = Experiment(default_config)
    experiment.run()


if __name__ == '__main__':
    main()
