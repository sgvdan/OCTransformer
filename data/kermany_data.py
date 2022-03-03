from pathlib import Path

import torch.nn.functional
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image

import util


class KermanyDataset(Dataset):
    def __init__(self, root, chosen_labels=[], transformations=None):
        super().__init__()
        self.transformations = transformations
        self.root = Path(root)

        self.classes = {label: label_value for label_value, label in enumerate(chosen_labels)}

        self.samples = []
        for label, label_value in self.classes.items():
            label_dir = self.root / label
            self.samples += [(img_path, label_value) for img_path in label_dir.iterdir()]

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = read_image(str(image_path))

        if self.transformations is not None:
            image = self.transformations(image)

        image = image.unsqueeze(dim=0).expand(-1, 3, -1, -1)

        return image, label

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return [labels for _, labels in self.samples]

    def get_classes(self):
        # TODO: REMOVE THIS! TREAT IT WITH LABEL NAME
        return self.classes


def get_kermany_transform(image_size):
    return transforms.Resize(image_size)


def setup_kermany(config):
    transform = get_kermany_transform(config.input_size)

    # Test Loader
    test_dataset = KermanyDataset(config.kermany_test_path, chosen_labels=config.kermany_labels,
                                  transformations=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size)

    # Eval Loader
    eval_dataset = KermanyDataset(config.kermany_eval_path, chosen_labels=config.kermany_labels,
                                  transformations=transform)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=config.batch_size)

    # Train Loader
    train_dataset = KermanyDataset(config.kermany_train_path, chosen_labels=config.kermany_labels,
                                   transformations=transform)
    train_weights = util.get_balance_weights(train_dataset.get_labels(), len(config.kermany_labels))
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                                               sampler=train_sampler)

    return train_loader, eval_loader, test_loader
