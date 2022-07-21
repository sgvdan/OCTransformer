# Default configuration to use
import torch

from util import dot_dict

default_config = dot_dict({'project': 'OCT-Transformer-paper',
                           # Logger
                           'log': True,
                           'log_group': 'ls-confidence-compr-sweep',
                           'log_frequency': 10,

                           # # Kermany Dataset
                           # 'dataset': 'kermany',
                           # 'kermany_train_path': '/home/projects/ronen/sgvdan/workspace/datasets/kermany/original/train',
                           # 'kermany_eval_path': '/home/projects/ronen/sgvdan/workspace/datasets/kermany/original/val',
                           # 'kermany_test_path': '/home/projects/ronen/sgvdan/workspace/datasets/kermany/original/test',
                           # 'input_size': (256, 256),
                           # 'num_slices': 1,
                           # 'batch_size': 1,
                           # 'labels': ['NORMAL', 'CNV', 'DME', 'DRUSEN'],

                           # Hadassah Dataset
                           'dataset': 'hadassah',
                           'hadassah_root': '/home/hsd/dannyh/data/oct/DR_TIFF',
                           'hadassah_dest': '/home/projects/ronen/sgvdan/workspace/datasets/hadassah/new/std-ls-confidence2',
                           'hadassah_annotations': '/home/projects/ronen/sgvdan/workspace/datasets/hadassah/new/std/std_annotations.xlsx',
                           'input_size': (256, 256),  # (496, 1024)
                           'num_slices': 37,

                           'batch_size': 1,
                           'labels': ['DR', 'DME', 'IRF'],

                           # Environment
                           'backbone': 'kermany_auto',  # imagenet_resnet18 / kermany_resnet18 / kermany_ls_resnet18 / kermany_ls11_resnet18 / resnet18
                           'model': 'vit',  # vit / slivernet / deepset / deepset2
                           'model_name': None,
                           'layer_segmentation_input': 'confidence-only',

                           # Models Bank
                           'keep_best_model': True,  # Whether to sync best model bank
                           'load_best_model': True,  # True to always load the best possible model, False for nominal

                           # Train Parameters
                           'optimizer': 'adam',
                           'criterion': 'binary_cross_entropy',  # cross_entropy
                           'scheduler': None,
                           'epochs': 10,
                           'lr': 1e-6,

                           # Model parameters
                           'attention_heads': 3,
                           'vit_depth': 12,
                           'embedding_dim': 768,

                            # Segmentation parameters
                           'gradcam_type': 'gradcam++',  # gradcam++, xgradcam, eigencam, fullgrad
                           'aug_smooth': True,
                           'eigen_smooth': False,
                           'attention_temperature': 0.05,
                           'mask_std_thresh': 3.5,

                           # General
                           'output_path': './output/',
                           'device': 'cuda',
                           })
