from functools import partial
from pathlib import Path
import torch
from PIL import Image
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode


class KermanyDataset(Dataset):
    def __init__(self, root, chosen_labels=[], transformations=None):
        super().__init__()
        self.transformations = transformations
        self.root = Path(root)

        self.classes = {label: label_value for label_value, label in enumerate(chosen_labels)}

        self.samples = []
        for label, label_value in self.classes.items():
            label_dir = self.root / label
            label_value = one_hot(torch.tensor(label_value), num_classes=len(self.classes))
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
        return torch.stack([label for _, label in self.samples])

    def get_classes(self):
        return self.classes


class KermanyLayerSegmentationDataset(KermanyDataset):
    def __init__(self, root, chosen_labels, transformations=None, layer_segmentation_transform=None):
        super().__init__(root, chosen_labels, transformations)
        self.layer_segmentation_transform = layer_segmentation_transform

    def __getitem__(self, idx):
        sample_data, label = super().__getitem__(idx)
        image_path, _ = self.samples[idx]

        ls_root = '/home/projects/ronen/sgvdan/workspace/datasets/kermany/layer-segmentation'
        layer_segmentation_path = (Path(ls_root) / '/'.join(image_path.parts[-3:])).with_suffix('.bmp')
        layer_segmentation_data = (transforms.ToTensor()(Image.open(str(layer_segmentation_path))) * 255).type(torch.uint8)

        layer_segmentation = self.layer_segmentation_transform(layer_segmentation_data)
        sample_data = torch.stack([layer_segmentation, sample_data[:, 0, :, :]], dim=1)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(sample_data[0, 0, :, :])
        # plt.figure()
        # plt.imshow(sample_data[0, 1, :, :])
        # plt.show()
        return sample_data, label


def get_kermany_transform(image_size):
    return transforms.Resize(image_size)


def get_layer_segementation_transform(image_size):
    return transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST)


def setup_kermany(config):
    if config.layer_segmentation_input:
        dataset = partial(KermanyLayerSegmentationDataset, chosen_labels=config.labels,
                          transformations=get_kermany_transform(config.input_size),
                          layer_segmentation_transform=get_layer_segementation_transform(config.input_size))
    else:
        dataset = partial(KermanyDataset, chosen_labels=config.labels,
                          transformations=get_kermany_transform(config.input_size))

    # Test Loader
    test_dataset = dataset(config.kermany_test_path)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size)

    # Eval Loader
    eval_dataset = dataset(config.kermany_eval_path)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=config.batch_size)

    # Train Loader
    train_dataset = dataset(config.kermany_train_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)

    return train_loader, eval_loader, test_loader
