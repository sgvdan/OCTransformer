# Default configuration to use
import torch

from util import dot_dict

default_config = dot_dict({'project': 'OCT-Transformer-paper',
                           # Logger
                           'log': False,
                           'log_group': 'temp',
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
                           'hadassah_root': '/home/nih/nih-dannyh/data/oct/DR_TIFF',
                           'hadassah_dest': '/home/projects/ronen/sgvdan/workspace/datasets/hadassah/new/std-masked',  # TODO: CHANGE TO STD-MASKED
                           'hadassah_ls_root': '/home/projects/ronen/sgvdan/workspace/datasets/hadassah/new/layer-segmentation',
                           'hadassah_annotations': '/home/projects/ronen/sgvdan/workspace/datasets/hadassah/new/std/std_annotations.xlsx',
                           'input_size': (256, 256),  # (496, 1024)
                           'num_slices': 11,
                           'batch_size': 1,
                           'labels': ['DR', 'DME', 'IRF'],

                           # Environment
                           'backbone': 'kermany_resnet18',  # imagenet_resnet18 / kermany_resnet18 / kermany_ls_resnet18 / kermany_ls11_resnet18 / resnet18
                           'model': 'vit',  # vit / slivernet / deepset
                           'model_name': '469YSOvo9b',  # vit-vanilla 57uyQrwq5Y
                           'layer_segmentation_input': False,

                           # Models Bank
                           'keep_best_model': True,  # Whether to sync best model bank
                           'load_best_model': True,  # True to always load the best possible model, False for nominal

                           # Train Parameters
                           'optimizer': 'adam',
                           'criterion': 'binary_cross_entropy',  # cross_entropy
                           'scheduler': None,
                           'epochs': 0,
                           'lr': 1e-5,

                           # Model parameters
                           'attention_heads': 4,
                           'vit_depth': 11,
                           'embedding_dim': 288,

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
