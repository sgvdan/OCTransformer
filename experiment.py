import wandb

import util
from data.hadassah_data import setup_hadassah
from data.kermany_data import setup_kermany
from models.bank import ModelsBank
from config import default_config
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
        self.train_loader, self.eval_loader, self.test_loader = self.setup_data()

        # Set up Model Bank
        self.model_bank = ModelsBank(self.config)

        # Set up Trainer
        self.trainer = Trainer(self.config, self.train_loader, self.eval_loader,
                               self.test_loader, self.model_bank, self.logger)

        # Set up Model Environment
        self.model, self.criterion, self.optimizer, self.scheduler = self.model_bank.get_environment()

    def setup_data(self):
        if self.config.dataset == 'hadassah':
            return setup_hadassah(self.config)
        elif self.config.dataset == 'kermany':
            return setup_kermany(self.config)
        else:
            raise NotImplementedError

    def run(self):
        self.trainer.train(self.model, self.criterion, self.optimizer, self.scheduler, self.config.epochs)

        if self.config.load_best_model:
            self.model_bank.load_best(self.model, self.optimizer, self.scheduler)  # Refresh model (avoid over fitting)

        accuracy = self.trainer.test(self.model)
        self.logger.log({'Overall_Accuracy': accuracy})


def main():
    experiment = Experiment(default_config)
    experiment.run()


if __name__ == '__main__':
    main()
