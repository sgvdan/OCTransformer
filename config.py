# Default configuration to use
from util import dot_dict

default_config = dot_dict({'project': 'OCTransformer',
                           # Logger
                           'log': True,
                           'log_group': 'preliminary',
                           'log_frequency': 10,

                           # Data
                           'train_cache': '37-slices/train',
                           'eval_cache': '37-slices/eval',
                           'test_cache': '37-slices/test',
                           'input_size': (496, 1024),  # Samples are not all same size

                           # Models Bank
                           'model_bank_open': True,
                           'backup_last_model': True,
                           'keep_best_model': True,

                           # Model
                           'embedding_dim': 768,

                           # Trainer
                           'epochs': 7,
                           'lr': 1e-4,
                           'batch_size': 5,

                           # General
                           'device': 'cuda',
                           })
