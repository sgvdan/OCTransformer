import config


class Experiment:
    def __init__(self, config):
        assert config is not None
        self.config = config

    def run(self):
        assert self.config is not None
        return 1


def main():
    experiment = Experiment(config.default_config)
    experiment.run()


if __name__ == '__main__':
    main()
