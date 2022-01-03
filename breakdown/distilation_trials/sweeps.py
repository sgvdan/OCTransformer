sweeps_config_resnet = {
    'name': 'resnet search',
    'method': 'bayes',
    'metric': {
        'goal': 'maximize',
        'name': 'Final Accuracy',
    },
    'parameters': {
        'pretrained_res': {
            'values': [True, False],
            'distribution': 'categorical',
        },
        'scheduler_step_size': {
            'max': 2,
            'min': 1,
            'distribution': 'int_uniform',
        },
        'scheduler_gamma': {
            'max': 1.997028076592444,
            'min': 0.06880986061232498,
            'distribution': 'uniform',
        },
        'weight_decay': {
            'max': 0.033569866313412654,
            'min': 0.00013422248069545946,
            'distribution': 'uniform',
        },
        'architecture': {
            'values': ['res18', 'resnet50', 'resnet101', 'resnet152'],
            'distribution': 'categorical',
        },
        'batch_size': {
            'max': 128,
            'min': 1,
            'distribution': 'int_uniform',
        },
        'scheduler': {
            'values': ['lr'],
            'distribution': 'categorical',
        },
        'optimizer': {
            'values': ['rmsprop', 'adam', 'sgd'],
            'distribution': 'categorical',
        },
        'epochs': {
            'max': 3,
            'min': 3,
            'distribution': 'int_uniform',
        },
        'seed': {
            'max': 19331.88309125071,
            'min': 354.1561390473186,
            'distribution': 'uniform',
        },
        'mom': {
            'max': 2.554618436876222,
            'min': 0.014827283535859812,
            'distribution': 'uniform',
        },
        'lr': {
            'max': 0.1,
            'min': 0.00001,
            'distribution': 'uniform'
        }
    }
}

sweep_config = {
    'method': 'random',  # grid, random
    'metric': {
        'name': 'val acc',
        'goal': 'maximize'
    },
    'parameters': {
        'architecture': {
            'values': ['vit', 'res18', 'resnet50', 'resnet101', 'resnet152', 'wide_resnet50_2', 'wide_resnet101_2'],
        },
        'pretrained_res': {
            'values': [True, False],
        },
        'seed': {
            'min': 1,
            'max': 10000,
            'distribution': 'uniform'
        },
        'lr': {
            'min': 0.00001,
            'max': 0.2,
            'distribution': 'uniform'
            # 'values': [1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5]
        },
        'epochs': {
            'values': [5]
        },
        'batch_size': {
            'values': [128, 64, 32, 2]
        },
        'optimizer': {
            'values': ['adam', 'sgd', 'rmsprop']
        },
        'mom': {
            'min': 0.001,
            'max': 1,
            'distribution': 'uniform'
        },
        'scheduler': {
            'values': ['lr']
        },
        'scheduler_gamma': {
            'min': 0.001,
            'max': 1,
            'distribution': 'uniform'
        },
        'scheduler_step_size': {
            'values': [1]
        },
        'vit_patch_size': {
            'values': [4, 6, 10, 16, 32, 64]
        },
        'vit_num_classes': {
            'values': [4]
        },
        'vit_embed_dim': {
            'values': [50, 100, 300, 768, 1024, 2048]
        },
        'vit_depth': {
            'values': [2, 4, 10, 12, 20, 32]
        },
        'vit_num_heads': {
            'values': [2, 4, 10, 12, 20, 32]
        },
        'vit_mlp_ratio': {
            'values': [2., 4., 10., 12.]
        },
        'vit_drop_rate': {
            'values': [0, 0.01, 0.1, 0.2, 0.3]
        },
        'vit_num_patches': {
            'values': [20, 50, 100, 300, 512]
        },
        'vit_attn_drop_rate': {
            'values': [0., 0.01, 0.1, 0.2, 0.2]
        },
        'weight_decay': {
            'min': 0.00001,
            'max': 0.01,
            'distribution': 'uniform'
        },
        # 'activation': {
        #     'values': ['relu', 'elu', 'selu', 'gelu']
        # }
    }
}
