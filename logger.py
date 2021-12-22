import torch
import wandb
import data
import numpy as np


class Logger:
    def __init__(self, config):
        self.config = config

        if self.config.log:
            wandb.login()
            wandb.init(project=self.config.project, group=self.config.log_group)

        self.steps, self.loss, self.pred, self.gt = None, None, None, None
        self.scratch()

    def get_current_accuracy(self, classes):
        accuracy, _ = self.assess(classes)
        return np.mean([value for value in accuracy.values()])

    def scratch(self):
        self.steps, self.loss, self.pred, self.gt = 0, 0, [], []

    def accumulate(self, pred, gt, loss=None):
        self.pred += pred.cpu()
        self.gt += gt.cpu()
        self.steps += 1

        if loss is not None:
            self.loss += loss

    def assess(self, classes):
        iou, accuracy = {}, {}
        _, pred_labels = torch.stack(self.pred).max(dim=1)
        gt_labels = torch.stack(self.gt)

        for label_name, label in classes.items():
            correct_predictions = ((pred_labels == gt_labels) & (gt_labels == label)).count_nonzero()
            incorrect_predictions = ((pred_labels == label) & (gt_labels != label)).count_nonzero()
            batch_size = (gt_labels == label).count_nonzero()

            iou[label_name] = correct_predictions / (batch_size + incorrect_predictions)
            accuracy[label_name] = correct_predictions / batch_size

        return accuracy, iou

    def log_train_periodic(self, epoch, classes):
        if not self.config.log:
            return

        if self.steps % self.config.log_frequency == 0:
            accuracy, iou = self.assess(classes)
            for (label, acc), (_, iou) in zip(accuracy.items(), iou.items()):
                wandb.log({'train/accuracy/{label}'.format(label=label): acc,
                           'train/iou/{label}'.format(label=label): iou,
                           'train/epoch': epoch})

            wandb.log({'train/loss': self.loss/self.steps})

    def log_eval_periodic(self, epoch, classes):
        if not self.config.log:
            return

        if self.steps % self.config.log_frequency == 0:
            accuracy, iou = self.assess(classes)
            for (label, acc), (_, iou) in zip(accuracy.items(), iou.items()):
                wandb.log({'evaluation/accuracy/{label}'.format(label=label): acc,
                           'evaluation/iou/{label}'.format(label=label): iou,
                           'evaluation/epoch': epoch})

    def log_test(self, classes):
        if not self.config.log:
            return

        accuracy, iou = self.assess(classes)
        for (label, acc), (_, iou) in zip(accuracy.items(), iou.items()):
            wandb.log({'test/accuracy/{label}'.format(label=label): acc,
                       'test/iou/{label}'.format(label=label): iou})
