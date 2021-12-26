# Default configuration to use
import torch

from util import dot_dict

default_config = dot_dict({'project': 'OCTransformer',
                           # Logger
                           'log': False,
                           'log_group': '1-slice',
                           'log_frequency': 10,

                           # Kermany Dataset
                           # 'dataset': 'kermany',
                           # 'train_cache': 'kermany/train',
                           # 'eval_cache': 'kermany/eval',
                           # 'test_cache': 'kermany/test',
                           # 'input_size': (256, 256),
                           # 'num_patches': 1,
                           # 'batch_size': 35,
                           # 'num_classes': 4,


                           # Hadassah Dataset
                           'dataset': 'hadassah',
                           'train_cache': '37-slices/train',
                           'eval_cache': '37-slices/eval',
                           'test_cache': '37-slices/test',
                           'input_size': (496, 1024),
                           'num_patches': 1,
                           'batch_size': 5,
                           'num_classes': 2,  # TODO: CHANGE THIS TO SOMEHOW TAKE IT FROM THE DATASET!!

                           # Models Bank
                           'environment': 'vit',
                           'model_name': 'KermanyViT',
                           'keep_best_model': True,  # Whether to sync best model bank
                           'load_best_model': True,  # True to always load the best possible model, False for nominal

                           # Model
                           'embedding_dim': 768,

                           # Trainer
                           'epochs': 10,
                           'lr': 1e-4,
                           'criterion': torch.nn.functional.cross_entropy,

                           # General
                           'device': 'cuda',
                           })
