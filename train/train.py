import numpy as np
import torch
from torch.nn.functional import one_hot
from tqdm import tqdm
from contextlib import ExitStack

from analysis.stats import get_performance_mesaures, sweep_thresholds_curves


class Trainer:
    def __init__(self, config, train_loader, eval_loader, test_loader, models_bank, logger):
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.models_bank = models_bank
        self.logger = logger

    def train(self, model, criterion, optimizer, scheduler, epochs):
        for epoch in range(epochs):
            tqdm.write("Epoch {epoch}/{epochs}".format(epoch=epoch + 1, epochs=epochs))

            # Train
            tqdm.write("\nTrain:")
            self.logger.scratch()
            for images, labels in tqdm(self.train_loader):
                pred, loss = self._feed_forward(model, images, labels, mode='train',
                                                criterion=criterion, optimizer=optimizer, scheduler=scheduler)
                self.logger.accumulate(pred=pred, gt=labels, loss=loss)
                self.logger.log_train_periodic(thres=model.thresholds)
            self.logger.log_train(epoch=epoch, thres=model.thresholds)

            # Update thresholds
            _, _, _, thresholds = sweep_thresholds_curves(*self.logger.get_accumulation(),
                                                          thres_range=np.arange(0.0, 1.01, 0.05))
            model.thresholds = torch.tensor(thresholds)

            # Evaluate
            tqdm.write("\nEvaluation:")
            self.logger.scratch()
            for images, labels in tqdm(self.eval_loader):
                pred, _ = self._feed_forward(model, images, labels, mode='eval')
                self.logger.accumulate(pred=pred, gt=labels)
            self.logger.log_eval(epoch=epoch, thres=model.thresholds)

            # Determine score & Sync Model Bank
            _, _, _, _, macro_f1 = get_performance_mesaures(*self.logger.get_accumulation(), model.thresholds)
            self.models_bank.sync_model(model, optimizer, scheduler, macro_f1)

    def test(self, model):
        tqdm.write("\nTest:")
        self.logger.scratch()
        for images, labels in tqdm(self.test_loader):
            pred, _ = self._feed_forward(model, images, labels, mode='eval')
            self.logger.accumulate(pred=pred, gt=labels)
        self.logger.log_test(model.thresholds)

        # Obtain Test Accuracy
        _, _, _, _, score = get_performance_mesaures(*self.logger.get_accumulation(), model.thresholds)
        tqdm.write("\nTest Score (Macro-F1): {}".format(score))

        return score

    def _feed_forward(self, model, images, labels, mode, criterion=None, optimizer=None, scheduler=None):
        # Make sure mode is as expected
        if mode == 'train' and not model.training:
            model.train()
            assert criterion is not None and optimizer is not None
        elif mode == 'eval' and model.training:
            model.eval()
            assert criterion is None and optimizer is None

        # Move to device
        images, labels = images.to(device=self.config.device, dtype=torch.float), \
                         labels.to(device=self.config.device, dtype=torch.float)

        # Run the model on the input batch
        with ExitStack() as stack:
            if not model.training:
                stack.enter_context(torch.no_grad())
            pred = model(images)

        # Calculate loss and update
        loss_value = 0.0

        if mode == 'train':
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()

            if scheduler is not None:
                scheduler.step(loss_value)

        return pred, loss_value
