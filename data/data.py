import torch
from torch.utils.data import Dataset
from data.cache import DatasetIterator


class E2EVolumeDataset(Dataset):
    def __init__(self, cache, transformations=None):
        self.cache = cache
        self.transformations = transformations

    def get_classes(self):
        return self.cache.get_classes()

    def get_labels(self):
        return self.cache.get_labels()

    def __iter__(self):
        return DatasetIterator(self)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        volume, *other = self.cache[idx]
        if self.transformations is not None:
            volume = self.transformations(volume)

        volume = volume.unsqueeze(dim=1).expand(-1, 3, -1, -1)
        return volume, *other


def get_balance_weights(dataset):
    classes = dataset.get_classes()
    labels = dataset.get_labels()
    num_classes = len(classes)
    num_scans = len(dataset)
    assert num_scans > 0 and num_classes > 0

    # Count # of appearances per each class
    count = [0] * num_classes
    for label in labels.values():  # TODO: If one_hot handling required - take `label.argmax(-1)`
        count[int(label)] += 1

    # Each class receives weight in reverse proportion to its # of appearances
    weight_per_class = [0.] * num_classes
    for idx in classes.values():
        weight_per_class[idx] = float(num_scans) / float(count[idx])

    # Assign class-corresponding weight for each element
    weights = [0] * num_scans
    for idx, label in labels.items():
        weights[idx] = weight_per_class[int(label)]

    return torch.FloatTensor(weights)
