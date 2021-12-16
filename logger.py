class Logger:
    def __init__(self, title, config):
        self.title = title
        self.config = config

    def log_train(self, pred, gt, loss):
        # TODO: LOG TO WANDB ACCURACY + IOU
        raise NotImplementedError

    def log_eval(self, pred, gt):
        # TODO: LOG TO WANDB ACCURACY + IOU
        raise NotImplementedError
