import os
from pathlib import Path

import numpy as np

import wandb

import util
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
        score = self.trainer.test(self.model)
        self.logger.log({'Overall_Score': score})

    def analyze(self):
        mix_dataset = MixedDataset(self.test_loader.dataset)
        mix_loader = torch.utils.data.DataLoader(dataset=mix_dataset, batch_size=self.config.batch_size)

        names, volumes, attns, cams, masks = [], [], [], [], []
        for idx, (volume, label) in enumerate(mix_loader):
            if label == 1:
                names.append('mix-' + str(idx))

                # Keep slices
                volumes.append(volume.squeeze(dim=0))  # Assume batch_size=1

                # Keep Attention Maps
                attns.append(self.model.get_attention_map(volume))  # Assume batch_size=1

                # Keep GradCam Maps
                cams.append(self.model.get_gradcam(volumes[-1]))

                # Keep thresholded weighted Attention x Gradcam masks
                masks.append(get_masks(attns[-1], cams[-1], std_thresh=self.config.mask_std_thresh))

                pred, _ = self.trainer._feed_forward(self.model, volume, label, mode='eval')
                print("MODEL'S PREDICTION:", pred)

        max_attn = np.stack(attns).max()
        max_cam = np.stack(cams).max()
        max_mask = np.stack(masks).max()

        for name, volume, attn, cam, mask in zip(names, volumes, attns, cams, masks):
            plot_slices(volume, logger=self.logger)
            plot_attention(attn / max_attn, logger=self.logger)
            plot_gradcam(volume, cam / max_cam, logger=self.logger)
            plot_masks(volume, mask / max_mask, logger=self.logger)

            self.logger.flush_images(name=name)


def main():
    experiment = Experiment(default_config)
    experiment.train()
    experiment.test()
    # experiment.analyze()


if __name__ == '__main__':
    main()
