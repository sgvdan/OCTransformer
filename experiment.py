import os
from pathlib import Path

import wandb

import util
from analysis.visualizer import plot_attention, plot_masks, plot_slices, plot_gradcam
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
        self.model, self.criterion, self.optimizer, self.scheduler = self.model_bank.get_environment()

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

        accuracy = self.trainer.test(self.model)
        self.logger.log({'Overall_Accuracy': accuracy})

    def analyze(self):
        mix_dataset = MixedDataset(self.test_loader.dataset)
        mix_loader = torch.utils.data.DataLoader(dataset=mix_dataset, batch_size=self.config.batch_size)

        for idx, (volume, label) in enumerate(self.test_loader):
            if label == 1:
                name = 'hadassah-{}'.format(idx)
                path = Path(self.config.output_path) / name
                os.makedirs(path, exist_ok=True)

                svolume = volume.squeeze(dim=0)  # Assume batch_size=1
                # Plot slices
                plot_slices(svolume, path, logger=self.logger)

                # Plot Attention
                attn = self.model.get_attention_map(volume)  # Assume batch_size=1
                plot_attention(attn, path, logger=self.logger)

                # Plot GradCam
                cam = self.model.get_gradcam(svolume)
                plot_gradcam(svolume, cam, path, logger=self.logger)

                # Plot thresholded weighted Attention x Gradcam
                plot_masks(svolume, attn, cam, path, logger=self.logger, std_thresh=self.config.mask_std_thresh)
                self.logger.flush_images(name=name)

                pred, _ = self.trainer._feed_forward(self.model, volume, label, mode='eval')
                print("MODEL'S PREDICTION:", pred)


def main():
    experiment = Experiment(default_config)
    experiment.train()
    experiment.analyze()


if __name__ == '__main__':
    main()
