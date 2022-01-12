import torch
from torch.utils.data import Dataset
from data.cache import DatasetIterator


class CachedDataset(Dataset):
    def __init__(self, cache):
        self.cache = cache

    def get_classes(self):
        return self.cache.get_classes()

    def get_labels(self):
        return self.cache.get_labels()

    def __iter__(self):
        return DatasetIterator(self)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]


def get_balance_weights(dataset, num_classes):
    labels = dataset.get_labels()
    num_scans = len(dataset)
    assert num_scans > 0 and num_classes > 0

    # Count # of appearances per each class
    count = [0] * num_classes
    for label in labels:  # TODO: If one_hot handling required - take `label.argmax(-1)`
        count[int(label)] += 1

    # Each class receives weight in reverse proportion to its # of appearances
    weight_per_class = [0.] * num_classes
    for idx in range(num_classes):
        weight_per_class[idx] = float(num_scans) / float(count[idx])

    # Assign class-corresponding weight for each element
    weights = [0] * num_scans
    for idx, label in enumerate(labels):
        weights[idx] = weight_per_class[int(label)]

    return torch.FloatTensor(weights)
