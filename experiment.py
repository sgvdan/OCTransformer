import torch
import util
from models.bank import ModelsBank
from models.vit import MyViT
from config import default_config
from data.cache import Cache
from data.data import get_balance_weights, CachedDataset
from data.hadassah.e2e_data import E2EVolumeDataset
from logger import Logger
from train.train import Trainer


class Experiment:
    def __init__(self, config):
        assert config is not None
        self.config = config

        util.make_deterministic()

        # Set up Logger
        self.logger = Logger(self.config)

        # Set up Data
        self.train_loader, self.eval_loader, self.test_loader = self.setup_data()

        # Set up Model Bank
        self.model_bank = ModelsBank(self.config)

        # Set up Trainer
        self.trainer = Trainer(self.config, self.train_loader, self.eval_loader,
                               self.test_loader, self.model_bank, self.logger)

        # Set up Model Environment
        self.model, self.optimizer = self.model_bank.get_environment(self.config.environment, self.config.model_name)
        self.criterion = self.config.criterion

    def setup_data(self):
        transform = util.get_default_transform(self.config.input_size)
        if self.config.dataset == 'hadassah':
            dataset = E2EVolumeDataset
        elif self.config.dataset == 'kermany':
            dataset = CachedDataset

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

        self.trainer.test(self.model)


def main():
    experiment = Experiment(default_config)
    experiment.run()


if __name__ == '__main__':
    main()
