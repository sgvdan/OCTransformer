import wandb
import data
import numpy as np


class Logger:
    def __init__(self, config):
        self.config = config

        if self.config.log:
            wandb.login()
            wandb.init(project=config.project)

        self.steps, self.loss, self.pred, self.gt = None, None, None, None
        self.scratch()

    def scratch(self):
        self.steps, self.loss, self.pred, self.gt = 0, 0, [], []

    def get_current_accuracy(self):
        accuracy, _ = self.assess()
        return np.mean([value for value in accuracy.values()])

    def assess(self):
        iou, accuracy = {}, {}

        for label_name, label in data.LABELS.items():
            correct_predictions = ((self.pred == self.gt) & (self.gt == label)).count_nonzero()
            incorrect_predictions = ((self.pred == label) & (self.gt != label)).count_nonzero()
            batch_size = (self.gt == label).count_nonzero()

            iou[label_name] = correct_predictions / batch_size
            accuracy[label_name] = correct_predictions / (batch_size + incorrect_predictions)

        return accuracy, iou

    def log_train(self, pred, gt, loss, epoch, periodic_flush=True):
        self.loss += loss
        self.pred += pred
        self.gt += gt
        self.steps += 1

        if not self.config.log:
            return

        if self.steps % self.config.log_frequency == 0 or not periodic_flush:
            accuracy, iou = self.asses()
            for (label, accuracy), (_, iou) in zip(accuracy, iou):
                wandb.log({'train/accuracy/{label}'.format(label): accuracy,
                           'train/iou/{label}'.format(label): iou,
                           'train/epoch': epoch})

            wandb.log({'train/loss': self.loss/self.steps})

    def log_eval(self, pred, gt, epoch, periodic_flush=True):
        self.steps += 1
        self.pred += pred
        self.gt += gt

        if not self.config.log:
            return

        if self.steps % self.config.log_frequency == 0 or not periodic_flush:
            accuracy, iou = self.asses()
            for (label, accuracy), (_, iou) in zip(accuracy, iou):
                wandb.log({'evaluation/accuracy/{label}'.format(label): accuracy,
                           'evaluation/iou/{label}'.format(label): iou,
                           'evaluation/epoch': epoch})

    def log_test(self, pred, gt):
        self.setps += 1
        self.pred += pred
        self.gt += gt

        if not self.config.log:
            return

        accuracy, iou = self.asses()
        for (label, accuracy), (_, iou) in zip(accuracy, iou):
            wandb.log({'test/accuracy/{label}'.format(label): accuracy,
                       'test/iou/{label}'.format(label): iou})
