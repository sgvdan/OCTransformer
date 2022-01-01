sweep_config = {
    'method': 'random',  # grid, random
    'metric': {
        'name': 'accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'values': 5
        },
        'batch_size': {
            'values': [256, 128, 64, 32, 2]
        },
        'dropout': {
            'values': [0.2,0.3, 0.4, 0.5]
        },
        'weight_decay': {
            'values': [0.0005, 0.005, 0.05]
        },
        'learning_rate': {
            'values': [1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5]
        },
        'optimizer': {
            'values': ['adam', 'sgd', 'rmsprop']
        },
        'activation': {
            'values': ['relu', 'elu', 'selu', 'gelu']
        }
    }
}
