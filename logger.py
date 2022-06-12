import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import wandb
from analysis.stats import get_performance_mesaures, sweep_thresholds_curves
from util import max_contour
from sklearn.metrics import auc

class Logger:
    def __init__(self, config):
        self.config = config

        self.steps, self.loss, self.pred, self.gt = None, None, None, None
        self.images = None

        self.scratch()
        self.train_gt = self.eval_gt = self.test_gt = None

    def log(self, data):
        if not self.config.log:
            return

        wandb.log(data)

    def get_accumulation(self):
        return torch.stack(self.pred), torch.stack(self.gt)

    def get_current_optimal_thresholds(self):
        pred = torch.stack(self.pred)
        gt = torch.stack(self.gt)
        thres_range = np.arange(0.0, 1.01, 0.05)

        _, _, _, opt_thres = sweep_thresholds_curves(pred, gt, thres_range=thres_range)

        return opt_thres

    def scratch(self):
        self.steps, self.loss, self.pred, self.gt = 0, 0, [], []
        self.images = []

    def accumulate(self, pred, gt, loss=None):
        self.pred += pred.detach().cpu()
        self.gt += gt.detach().cpu()
        self.steps += 1

        if loss is not None:
            self.loss += loss

    def log_stats(self, title, epoch, thres=None):
        pred = torch.stack(self.pred)
        gt = torch.stack(self.gt)

        if thres is None:
            thres = torch.tensor([0.5] * len(self.config.labels))

        accuracy, precision, recall, f1, macro_f1 = get_performance_mesaures(pred, gt, thres)

        for idx, label in enumerate(self.config.labels):
            self.log({'{title}/accuracy/{label}'.format(title=title, label=label): accuracy[idx],
                      '{title}/precision/{label}'.format(title=title, label=label): precision[idx],
                      '{title}/recall/{label}'.format(title=title, label=label): recall[idx],
                      '{title}/f1/{label}'.format(title=title, label=label): f1[idx],
                      '{title}/epoch'.format(title=title): epoch})

        self.log({'{title}/macro_f1'.format(title=title): macro_f1,
                  '{title}/epoch'.format(title=title): epoch})

    def log_train_periodic(self, thres):
        if self.steps % self.config.log_frequency == 0:
            self.log_train(None, thres)

    def log_train(self, epoch, thres):
        self.train_gt = self.gt

        if not self.config.log:
            return

        self.log({'train/loss': self.loss/self.steps})
        self.log_stats(title='train', epoch=epoch, thres=thres)

    def log_eval(self, epoch, thres):
        self.eval_gt = self.gt

        if not self.config.log:
            return

        self.log_stats(title='evaluation', epoch=epoch, thres=thres)

    def log_test(self, thres):
        self.test_gt = self.gt

        if not self.config.log:
            return

        self.log_stats(title='test', epoch=None, thres=thres)

    def log_curves(self):
        if not self.config.log:
            return

        pred = torch.stack(self.pred)
        gt = torch.stack(self.gt)
        thres_range = np.arange(0.0, 1.01, 0.05)
        pr, roc, f1, thresholds = sweep_thresholds_curves(pred, gt, thres_range=thres_range)

        pr_xs, roc_xs = [], []
        pr_ys, roc_ys = [], []
        mean_pr_auc, mean_roc_auc = 0, 0
        for idx, label in enumerate(self.config.labels):
            # Log details
            roc_details = wandb.Table(data=np.concatenate([thres_range[:, None], roc[:, :, idx], f1[:, idx, None]], axis=1),
                                      columns=["Threshold", "False Positive Rate", "True Positive Rate", "F1-Score"])
            pr_details = wandb.Table(data=np.concatenate([thres_range[:, None], pr[:, :, idx], f1[:, idx, None]], axis=1),
                                     columns=["Threshold", "Recall", "Precision", "F1-Score"])
            wandb.log({'roc-details-' + label: roc_details, 'pr-details-' + label: pr_details})

            # Log graphs (obtain singular (max) y per x
            xs, ys = max_contour(pr[:, :, idx])
            area_under_curve = auc(np.array(xs), np.array(ys))
            pr_xs.append(xs)
            pr_ys.append(ys)
            wandb.log({'pr-auc-' + label: area_under_curve})
            mean_pr_auc += area_under_curve / pr.shape[2]

            xs, ys = max_contour(roc[:, :, idx])
            area_under_curve = auc(np.array(xs), np.array(ys))
            roc_xs.append(xs)
            roc_ys.append(ys)
            wandb.log({'roc-auc-' + label: area_under_curve})
            mean_roc_auc += area_under_curve / roc.shape[2]

        # Print ideal thresholds
            tqdm.write(label + ' - optimal threshold:' + str(round(thresholds[idx], 2)))

        wandb.log({'pr-graph': wandb.plot.line_series(pr_xs, pr_ys, keys=self.config.labels,
                                                      title="Precision-Recall Curve"),
                   'roc-graph': wandb.plot.line_series(roc_xs, roc_ys, keys=self.config.labels,
                                                       title="Receiver Operating Characteristic Curve"),
                   'mean-pr-auc': mean_pr_auc,
                   'mean-roc-auc': mean_roc_auc})

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

    def log_summary(self, key, val):
        if not self.config.log:
            return

        wandb.summary[key] = val
