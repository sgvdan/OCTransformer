# Default configuration to use
import torch

from util import dot_dict

default_config = dot_dict({'project': 'OCTransformer',
                           # Logger
                           'log': False,
                           'log_group': 'misc',
                           'log_frequency': 10,

                           # # Kermany Dataset
                           # 'dataset': 'kermany',
                           # 'kermany_train_path': '/home/projects/ronen/sgvdan/workspace/datasets/kermany/train',
                           # 'kermany_eval_path': '/home/projects/ronen/sgvdan/workspace/datasets/kermany/val',
                           # 'kermany_test_path': '/home/projects/ronen/sgvdan/workspace/datasets/kermany/test',
                           # 'kermany_labels': ['NORMAL', 'CNV', 'DME', 'DRUSEN'],
                           # 'input_size': (256, 256),
                           # 'num_slices': 1,
                           # 'batch_size': 1,
                           # 'num_classes': 4,  # TODO: CHANGE THIS TO len(kermany_labels)

                           # Hadassah Dataset
                           'dataset': 'hadassah',
                           'input_size': (256, 256),  # (496, 1024)
                           'num_slices': 5,
                           'batch_size': 1,
                           'num_classes': 2,  # TODO: CHANGE THIS TO SOMEHOW TAKE IT FROM THE DATASET!!

                           # Environment
                           'backbone': 'kermany_resnet18',  # imagenet_resnet18 / kermany_resnet18 / resnet18
                           'model': 'vit',
                           'model_name': None,

                           # Models Bank
                           'keep_best_model': True,  # Whether to sync best model bank
                           'load_best_model': True,  # True to always load the best possible model, False for nominal

                           # Train Parameters
                           'optimizer': 'adam',
                           'criterion': 'cross_entropy',
                           'scheduler': None,
                           'epochs': 3,
                           'lr': 1e-5,

                           'train_size': 0.65,
                           'eval_size': 0.1,
                           'test_size': 0.25,

                           # Model
                           'attention_heads': 12,
                           'vit_depth': 12,
                           'embedding_dim': 768,

                           # General
                           'output_path': './output/',
                           'device': 'cuda',
                           })
