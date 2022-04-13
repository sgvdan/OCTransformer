import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import wandb
from analysis.stats import get_performance_mesaures, sweep_thresholds_curves
from util import max_contour


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

    def get_current_optimal_thresholds(self):
        pred = torch.stack(self.pred)
        gt = torch.stack(self.gt)
        thres_range = np.arange(0.0, 1.01, 0.05)

        _, _, _, opt_thres = sweep_thresholds_curves(pred, gt, thres_range=thres_range)

        return opt_thres

    def get_current_macro_f1(self):
        pred = torch.stack(self.pred)
        gt = torch.stack(self.gt)
        thres = torch.tensor(self.config.confidence_thresholds)
        _, _, _, _, macro_f1 = get_performance_mesaures(pred, gt, thres)

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

        accuracy, precision, recall, f1, macro_f1 = get_performance_mesaures(pred, gt, thres)

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
        thres_range = np.arange(0.0, 1.01, 0.05)
        pr, roc, f1, opt_thres = sweep_thresholds_curves(pred, gt, thres_range=thres_range)

        pr_xs, roc_xs = [], []
        pr_ys, roc_ys = [], []
        for idx, label in enumerate(self.config.labels):
            # Log details
            roc_details = wandb.Table(data=np.concatenate([thres_range[:, None], roc[:, :, idx], f1[:, idx, None]], axis=1),
                                      columns=["Threshold", "False Positive Rate", "True Positive Rate", "F1-Score"])
            pr_details = wandb.Table(data=np.concatenate([thres_range[:, None], pr[:, :, idx], f1[:, idx, None]], axis=1),
                                     columns=["Threshold", "Recall", "Precision", "F1-Score"])
            wandb.log({'roc-details-' + label: roc_details, 'pr-details-' + label: pr_details})

            # Log graphs (obtain singular (max) y per x
            xs, ys = max_contour(pr[:, :, idx])
            pr_xs.append(xs)
            pr_ys.append(ys)

            xs, ys = max_contour(roc[:, :, idx])
            roc_xs.append(xs)
            roc_ys.append(ys)

            # Print ideal thresholds
            tqdm.write(label + ' - optimal threshold:' + str(round(opt_thres[idx], 2)))

        wandb.log({'pr-graph': wandb.plot.line_series(pr_xs, pr_ys, keys=self.config.labels,
                                                      title="Precision-Recall Curve"),
                   'roc-graph': wandb.plot.line_series(roc_xs, roc_ys, keys=self.config.labels,
                                                       title="Receiver Operating Characteristic Curve")})

    def log_image(self, image, caption):
        self.images.append((image, caption))

    def flush_images(self, name):
        path = Path(self.config.output_path) / name
        os.makedirs(path, exist_ok=True)

        wandb_packet = []

        for img, caption in self.images:
            img.save(path / (caption + '.png'))
            wandb_packet.append(wandb.Image(img, caption=caption))

        self.log({('images/' + name): wandb_packet})
        self.images = []

