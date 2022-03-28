import numpy as np
import wandb

import util
from analysis.stats import get_binary_prediction
from analysis.visualizer import plot_attention, plot_masks, plot_slices, plot_gradcam, get_masks
from data.hadassah_data import setup_hadassah
from data.hadassah_mix import MixedDataset
from data.kermany_data import setup_kermany
from models.bank import ModelsBank
from config import default_config
from logger import Logger
from train.train import Trainer
import torch


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
        self.model, self.optimizer, self.scheduler = self.model_bank.get_environment()
        self.criterion = self.model_bank.get_balanced_criterion(self.train_loader)

    def setup_data(self):
        if self.config.dataset == 'hadassah':
            return setup_hadassah(self.config)
        elif self.config.dataset == 'kermany':
            return setup_kermany(self.config)
        else:
            raise NotImplementedError

    def train(self):
        self.trainer.train(self.model, self.criterion, self.optimizer, self.scheduler, self.config.epochs)

        if self.config.load_best_model:
            self.model_bank.load_best(self.model, self.optimizer, self.scheduler)  # Refresh model (avoid over fitting)

    def test(self):
        return self.trainer.test(self.model)

    def analyze(self):
        mix_dataset = MixedDataset(self.test_loader.dataset)
        mix_loader = torch.utils.data.DataLoader(dataset=mix_dataset, batch_size=self.config.batch_size)

        shuffle_test = torch.utils.data.DataLoader(dataset=self.test_loader.dataset,
                                                   batch_size=self.config.batch_size,
                                                   shuffle=True)

        count = 0
        for idx, (volume, label) in enumerate(shuffle_test):
            if label.sum() == 0:
                continue

            count += 1
            if count > 10:
                break

            # Keep slices
            plot_slices(volume.squeeze(dim=0), logger=self.logger, title='raw')

            # Keep Attention Maps
            attn = self.model.get_attention_map(volume)
            plot_attention(attn, logger=self.logger, title='attention')

            # Generate Weighted GradCam Masks per each positive label
            pred, _ = self.trainer._feed_forward(self.model, volume, label, mode='eval')
            binary_pred = get_binary_prediction(pred.cpu(), torch.tensor(self.config.confidence_thresholds))
            target_labels = binary_pred.nonzero()[:, 1].tolist()  # gather all positive labels

            if target_labels:
                label = '_'.join([self.config.labels[category] for category in target_labels])
                cam = self.model.get_gradcam(volume)
                plot_gradcam(volume.squeeze(dim=0), cam, logger=self.logger, title='gradcam-' + label)

                mask = get_masks(attn, cam, std_thresh=self.config.mask_std_thresh)
                plot_masks(volume.squeeze(dim=0), mask, logger=self.logger, title='mask-' + label)

            self.logger.flush_images(name='temp-' + str(idx))
            print("Model's {} prediction:".format(idx), [self.config.labels[idx] for idx in target_labels])


def main():
    experiment = Experiment(default_config)
    experiment.train()
    experiment.test()
    # experiment.analyze()


if __name__ == '__main__':
    main()
