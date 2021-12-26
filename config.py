# Default configuration to use
import torch

from util import dot_dict

default_config = dot_dict({'project': 'OCTransformer',
                           # Logger
                           'log': False,
                           'log_group': 'kermany',
                           'log_frequency': 10,

                           # Data
                           'dataset': 'hadassah',  # kermany
                           'train_cache': '37-slices/train',  # kermany/train
                           'eval_cache': '37-slices/eval',  # kermany/eval
                           'test_cache': '37-slices/test',  #kermany/test
                           'input_size': (496, 1024),  # hadassah: (496, 1024), kermany: (256,256) Samples are not all same size

                           # Models Bank
                           'environment': 'vit',
                           'model_name': 'MyViT',
                           'keep_best_model': True,  # Whether to sync best model bank
                           'load_best_model': True,  # True to always load the best possible model, False for nominal

                           # Model
                           'embedding_dim': 768,
                           'num_patches': 37,

                           # Trainer
                           'epochs': 10,
                           'lr': 1e-4,
                           'batch_size': 5,
                           'criterion': torch.nn.functional.cross_entropy,

                           # General
                           'device': 'cuda',
                           })
