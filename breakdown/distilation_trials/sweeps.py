sweep_config = {
    'method': 'random',  # grid, random
    'metric': {
        'name': 'val acc',
        'goal': 'maximize'
    },
    'parameters': {
        'architecture': {
            'values': ['vit', 'res'],
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
            'values': 5
        },
        'batch_size': {
            'values': [2048, 1024, 512, 256, 128, 64, 32, 2]
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
            'values': 'lr'
        },
        'scheduler_gamma': {
            'min': 0.001,
            'max': 1,
            'distribution': 'uniform'
        },
        'scheduler_step_size': {
            'values': 1
        },
        'vit_patch_size': {
            'values': [4, 6, 10, 16, 32, 64]
        },
        'vit_num_classes': {
            'values': 4
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
