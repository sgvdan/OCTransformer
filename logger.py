import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import wandb
from analysis.stats import get_stats


class Logger:
    def __init__(self, config):
        self.config = config

        self.steps, self.loss, self.pred, self.gt = None, None, None, None
        self.images = None

        self.scratch()

    def log(self, data):
        if not self.config.log:
            return

        wandb.log(data)

    def get_current_macro_f1(self):
        pred = torch.stack(self.pred)
        gt = torch.stack(self.gt)
        thres = torch.tensor(self.config.confidence_thresholds)
        _, _, _, _, macro_f1 = get_stats(pred, gt, thres)

        return macro_f1

    def scratch(self):
        self.steps, self.loss, self.pred, self.gt = 0, 0, [], []
        self.images = []

    def accumulate(self, pred, gt, loss=None):
        self.pred += pred.detach().cpu()
        self.gt += gt.detach().cpu()
        self.steps += 1

        if loss is not None:
            self.loss += loss

    def log_stats(self, title, epoch):
        pred = torch.stack(self.pred)
        gt = torch.stack(self.gt)
        thres = torch.tensor(self.config.confidence_thresholds)

        accuracy, precision, recall, f1, macro_f1 = get_stats(pred, gt, thres)

        for idx, label in enumerate(self.config.labels):
            self.log({'{title}/accuracy/{label}'.format(title=title, label=label): accuracy[idx],
                      '{title}/precision/{label}'.format(title=title, label=label): precision[idx],
                      '{title}/recall/{label}'.format(title=title, label=label): recall[idx],
                      '{title}/f1/{label}'.format(title=title, label=label): f1[idx],
                      '{title}/epoch'.format(title=title): epoch})

        self.log({'{title}/macro_f1'.format(title=title): macro_f1,
                  '{title}/epoch'.format(title=title): epoch})

    def log_train_periodic(self):
        if self.steps % self.config.log_frequency == 0:
            self.log_train(None)

    def log_train(self, epoch):
        if not self.config.log:
            return

        self.log({'train/loss': self.loss/self.steps})
        self.log_stats(title='train', epoch=epoch)

    def log_eval(self, epoch):
        if not self.config.log:
            return

        self.log_stats(title='evaluation', epoch=epoch)

    def log_test(self):
        if not self.config.log:
            return

        self.log_stats(title='test', epoch=None)
        self.log_curves()

    def log_curves(self):
        pred = torch.stack(self.pred)
        gt = torch.stack(self.gt)

        tqdm.write("\nLogging curves:")
        for idx, label in tqdm(enumerate(self.config.labels)):
            for thres in np.arange(0.0, 1.0, 0.1):
                _gt = gt[:, idx]  # W&B SHIT  TODO: sort this out
                _pred = (pred[:, idx] > thres).to(dtype=torch.int64)
                _pred = torch.nn.functional.one_hot(_pred, num_classes=2)

                self.log({'roc-' + label: wandb.plot.roc_curve(_gt, _pred, title=('ROC-' + label)),
                          'pr-' + label: wandb.plot.pr_curve(_gt, _pred, title=('PR-' + label))})

    def log_image(self, image, caption):
        if not self.config.log:
            return

        self.images.append((image, caption))

    def flush_images(self, name):
        if not self.config.log:
            return

        path = Path(self.config.output_path) / name
        os.makedirs(path, exist_ok=True)

        wandb_packet = []

        for img, caption in self.images:
            img.save(path / (caption + '.png'))
            wandb_packet.append(wandb.Image(img, caption=caption))

        self.log({('images/' + name): wandb_packet})

        self.images = []

