# Default configuration to use
import torch

from util import dot_dict

default_config = dot_dict({'project': 'OCTransformer',
                           # Logger
                           'log': True,
                           'log_group': 'kermany-prelim',
                           'log_frequency': 30,

                           # Kermany Dataset
                           'dataset': 'kermany',
                           'kermany_train_path': '/home/projects/ronen/sgvdan/workspace/datasets/kermany/train',
                           'kermany_eval_path': '/home/projects/ronen/sgvdan/workspace/datasets/kermany/val',
                           'kermany_test_path': '/home/projects/ronen/sgvdan/workspace/datasets/kermany/test',
                           'labels': ['NORMAL', 'CNV', 'DME', 'DRUSEN'],
                           'input_size': (256, 256),
                           'num_slices': 1,
                           'batch_size': 1,
                           'confidence_thresholds': [0.05, 0.70, 0.50, 0.75],

                           # # Hadassah Dataset
                           # 'dataset': 'hadassah',
                           # 'input_size': (256, 256),  # (496, 1024)
                           # 'num_slices': 5,
                           # 'batch_size': 1,
                           # 'labels': ['DR', 'DME', 'IRF', 'SRF', 'ELLIPSOID ZONE DISRUPTION '],
                           # 'confidence_thresholds': [0.15, 0.95, 0.95, 0.40, 0.05],  # Obtained from https://wandb.ai/sgvdan/OCTransformer/runs/pwg2qn0h

                           # Environment
                           'backbone': 'kermany_resnet18',  # imagenet_resnet18 / kermany_resnet18 / resnet18
                           'model': 'vit',
                           'model_name': 'hnCZEGBOIx',

                           # Models Bank
                           'keep_best_model': True,  # Whether to sync best model bank
                           'load_best_model': True,  # True to always load the best possible model, False for nominal

                           # Train Parameters
                           'optimizer': 'adam',
                           'criterion': 'binary_cross_entropy',  # cross_entropy
                           'scheduler': None,
                           'epochs': 0,#6,
                           'lr': 1e-5,

                           'train_size': 0.65,
                           'eval_size': 0.1,
                           'test_size': 0.25,

                           # Model parameters
                           'attention_heads': 1,
                           'vit_depth': 1,
                           'embedding_dim': 768,

                            # Segmentation parameters
                           'attention_temperature': 0.1,
                           'mask_std_thresh': 3,

                           # General
                           'output_path': './output/',
                           'device': 'cuda',
                           })
