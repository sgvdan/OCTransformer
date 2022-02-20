import torch
from torch.nn.functional import one_hot
from tqdm import tqdm
from contextlib import ExitStack


class Trainer:
    def __init__(self, config, train_loader, eval_loader, test_loader, models_bank, logger):
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.config = config
        self.models_bank = models_bank
        self.logger = logger

    def train(self, model, criterion, optimizer, scheduler, epochs):
        for epoch in range(epochs):
            tqdm.write("Epoch {epoch}/{epochs}".format(epoch=epoch+1, epochs=epochs))

            # Train
            tqdm.write("\nTrain:")
            self.logger.scratch()
            for images, labels in tqdm(self.train_loader):
                pred, loss = self._feed_forward(model, images, labels, mode='train',
                                                criterion=criterion, optimizer=optimizer, scheduler=scheduler)
                self.logger.accumulate(pred=pred, gt=labels, loss=loss)
                self.logger.log_train_periodic(classes=self.train_loader.dataset.get_classes())
            self.logger.log_train(epoch=epoch, classes=self.train_loader.dataset.get_classes())

            # Evaluate
            tqdm.write("\nEvaluation:")
            self.logger.scratch()
            for images, labels in tqdm(self.eval_loader):
                pred, _ = self._feed_forward(model, images, labels, mode='eval')
                self.logger.accumulate(pred=pred, gt=labels)
            self.logger.log_eval(epoch=epoch, classes=self.eval_loader.dataset.get_classes())

            # Sync Model Bank
            accuracy = self.logger.get_current_accuracy(classes=self.eval_loader.dataset.get_classes())
            self.models_bank.sync_model(model, optimizer, scheduler, accuracy)

    def test(self, model):
        tqdm.write("\nTest:")
        self.logger.scratch()
        for images, labels in tqdm(self.test_loader):
            pred, _ = self._feed_forward(model, images, labels, mode='eval')
            self.logger.accumulate(pred=pred, gt=labels)
        self.logger.log_test(classes=self.test_loader.dataset.get_classes())

        # Obtain Test Accuracy
        accuracy = self.logger.get_current_accuracy(classes=self.test_loader.dataset.get_classes())
        tqdm.write("\nTest Accuracy: {}".format(accuracy))

        return accuracy

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
                         labels.to(device=self.config.device, dtype=torch.int64)

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
