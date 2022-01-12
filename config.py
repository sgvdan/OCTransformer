# Default configuration to use
import torch

from util import dot_dict

default_config = dot_dict({'project': 'OCTransformer',
                           # Logger
                           'log': False,
                           'log_group': 'hadassah/dme-classification',
                           'log_frequency': 10,

                           # Kermany Dataset
                           # 'dataset': 'kermany',
                           # 'train_cache': 'kermany/train',
                           # 'eval_cache': 'kermany/eval',
                           # 'test_cache': 'kermany/test',
                           # 'input_size': (256, 256),
                           # 'num_slices': 1,
                           # 'batch_size': 35,
                           # 'num_classes': 4,

                           # Hadassah Dataset
                           # 'dataset': 'hadassah',
                           # 'train_cache': '37-slices/train',
                           # 'eval_cache': '37-slices/eval',
                           # 'test_cache': '37-slices/test',

                           'records_path': '/home/projects/ronen/sgvdan/workspace/projects/OCTransformer/datasets/'
                                           'full-37/records.pkl',
                           'input_size': (496, 1024),
                           'num_slices': 37,
                           'batch_size': 5,
                           'num_classes': 2,  # TODO: CHANGE THIS TO SOMEHOW TAKE IT FROM THE DATASET!!

                           # Environment
                           'model': 'vit',
                           'model_name': None,

                           # Models Bank
                           'keep_best_model': True,  # Whether to sync best model bank
                           'load_best_model': True,  # True to always load the best possible model, False for nominal

                           # Train Parameters
                           'optimizer': 'adam',
                           'criterion': 'cross_entropy',
                           'epochs': 5,
                           'lr': 1e-4,

                           # Model
                           'embedding_dim': 768,

                           # General
                           'device': 'cuda',
                           })
