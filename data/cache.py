import os
import pickle
from pathlib import Path


class Cache:
    def __init__(self, name):
        """
        :param name: Unique name for this cache
        """
        self.name = name
        self.cache_path = Path('.cache') / self.name
        self.cache_fs_path = self.cache_path / '.cache_fs'
        self.labels_path = self.cache_path / '.labels'
        self.classes_path = self.cache_path / '.classes'

        # Create cache directory
        if not self.cache_path.exists():
            self.cache_path.mkdir(parents=True)

        # Retrieve file-system dictionary
        if self.cache_fs_path.exists():
            with open(self.cache_fs_path, 'rb') as file:
                self.cache_fs = pickle.load(file)
        else:
            self.cache_fs = {}

        # Retrieve labels
        if self.labels_path.exists():
            with open(self.labels_path, 'rb') as file:
                self.labels = pickle.load(file)
        else:
            self.labels = {}

        # Retrieve classes
        if self.classes_path.exists():
            with open(self.classes_path, 'rb') as file:
                self.classes = pickle.load(file)
        else:
            self.classes = {}

    def set_class(self, key, value):
        self.classes[key] = value

        with open(self.classes_path, 'wb+') as file:
            pickle.dump(self.classes, file)

    def get_classes(self):
        return self.classes

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        """
        :param idx: idx of data
        :return: returns the data stored in cache's idx
        """

        assert idx < len(self)

        with open(self.cache_fs[idx], 'rb') as file:
            data = pickle.load(file)

        return data, self.labels[idx]

    def __setitem__(self, idx, value):
        """
        :param idx: index to set by
        :param value: (data, label) - tuple of data and label, both should be torch tensors
        :return: None
        """
        data, label = value
        assert label in self.classes.values()

        item_path = self.cache_path / str(idx)
        with open(item_path, 'wb+') as file:
            pickle.dump(data, file)

        self.cache_fs[idx] = item_path
        self.labels[idx] = label

        with open(self.cache_fs_path, 'wb+') as file:
            pickle.dump(self.cache_fs, file)

        with open(self.labels_path, 'wb+') as file:
            pickle.dump(self.labels, file)

    def __len__(self):
        return len(self.cache_fs)

    def append(self, value):
        self[len(self)] = value


class PartialCache:
    def __init__(self, cache, lut):
        self.cache = cache
        self.lut = lut

    def get_labels(self):
        return {idx: self.cache.labels[lut_idx] for idx, lut_idx in enumerate(self.lut)}

    def __getitem__(self, idx):
        return self.cache[self.lut[idx]]

    def __setitem__(self, idx, value):
        self.cache[self.lut[idx]] = value

    def __len__(self):
        return len(self.lut)


class DatasetIterator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.curr_idx = 0

    def __iter__(self):
        self.curr_idx = 0
        return self

    def __next__(self):
        if self.curr_idx == len(self.dataset):
            raise StopIteration

        self.curr_idx += 1
        return self.dataset[self.curr_idx - 1]
