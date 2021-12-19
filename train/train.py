import os
from tqdm import tqdm


class Trainer:
    def __init(self, config, train_loader, evaluation_loader, test_loader, models_bank, logger):
        self.config = config
        self.train_loader = train_loader
        self.evaluation_loader = evaluation_loader
        self.test_loader = test_loader
        self.config = config
        self.models_bank = models_bank
        self.logger = logger

    def train(self, model, criterion, optimizer, epochs):
        for epoch in range(epochs):
            tqdm.write("Epoch {epoch}/{epochs}".format(epoch=epoch, epochs=epochs))

            # Train
            tqdm.write("\nTrain:")
            self.logger.scratch()
            for images, labels in tqdm(self.train_loader):
                pred, loss = self._feed_forward(model, images, labels, mode='train',
                                                criterion=criterion, optimizer=optimizer)
                self.logger.log_train(pred=pred, gt=labels, loss=loss, epoch=epoch, periodic_flush=True)

            # Evaluate
            tqdm.write("\nEvaluation:")
            self.logger.scratch()
            for images, labels in tqdm(self.evaluation_loader):
                pred, _ = self._feed_forward(model, images, labels, mode='eval')
                self.logger.log_eval(pred=pred, gt=labels, epoch=epoch, periodic_flush=True)

            # Sync Model Bank
            accuracy = self.logger.get_current_accuracy()
            self.bank.sync_model(model, optimizer, epoch, accuracy)

    def test(self, model):
        tqdm.write("\nTest:")
        self.logger.scratch()
        for images, labels in tqdm(self.test_loader):
            pred, _ = self._feed_forward(model, images, labels, mode='eval')

        self.logger.log_test(pred=pred, gt=labels, periodic_flush=False)


    @staticmethod
    def _feed_forward(model, images, labels, mode, criterion=None, optimizer=None):
        # Make sure mode is as expected
        if mode == "train" and not model.training:
            model.train()
            assert criterion is not None and optimizer is not None
        elif mode == "eval" and model.training:
            model.eval()
            assert criterion is None and optimizer is None

        # Move to device
        images, labels = images.to(device=model.device), labels.to(device=model.device)

        # Run the model on the input batch
        pred = model(images)

        # Calculate loss and update
        loss_value = 0.0

        if mode == 'train':
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()

        return pred, loss_value
