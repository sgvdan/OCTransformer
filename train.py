import os

from tqdm import tqdm


class Trainer:
    def __init(self, config, train_loader, evaluation_loader, models_bank, logger):
        self.config = config
        self.train_loader = train_loader
        self.evaluation_loader = evaluation_loader
        self.config = config
        self.models_bank = models_bank
        self.logger = logger

    def train(self, model, criterion, optimizer, epochs):
        for epoch in range(epochs):
            tqdm.write("Epoch {epoch}/{epochs}\n Train:".format(epoch=epoch, epochs=epochs))
            self.logger.scratch()

            for images, labels in tqdm(self.train_loader):
                pred, loss = self._feed_forward(model, images, labels, mode='train',
                                                criterion=criterion, optimizer=optimizer)
                self.logger.log_train(pred=pred, gt=labels, loss=loss)

            tqdm.write("\nEvaluation:")
            for images, labels in tqdm(self.evaluation_loader):
                pred, _ = self._feed_forward(model, images, labels, mode='eval')
                self.logger.log_eval(pred=pred, gt=labels)

            avg_accuracy = self.logger.get_evaluation_accuracy()

            self.bank.sync_model(model, optimizer, epoch, avg_accuracy)

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
