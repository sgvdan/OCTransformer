import torch

from utils import dot_dict

default_config = dot_dict({'project': 'OCTransformer',

                           # Kermany Dataset
                           'dataset': 'kermany',
                           'input_size': (256, 256),
                           'num_slices': 1,
                           'batch_size': 35,
                           'num_classes': 4,

                           # Environment
                           'model': 'vit',
                           'model_name': None,

                           # Models Bank
                           'keep_best_model': True,  # Whether to sync best model bank
                           'load_best_model': True,  # True to always load the best possible model, False for nominal

                           # Train Parameters
                           'optimizer': 'adam',
                           'criterion': 'cross_entropy',
                           'epochs': 10,
                           'lr': 1e-4,

                           # Model
                           'embedding_dim': 768,

                           # General
                           'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                           })