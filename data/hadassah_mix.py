# A dataset of synthetica volumes - composed of mix of healthy/sick slices, to evaluate the attention mechanism
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class MixedDataset(Dataset):
    def __init__(self, dataset, count=100, slices_num=5):
        self.dataset = dataset
        self.samples = []

        self.len = count
        self.slices_num = slices_num

        self.healthy_samples = []
        self.sick_samples = []

        self.generate_mixed_samples()
        self.label = torch.tensor(1)

    def generate_mixed_samples(self):
        for idx, (sample_data, sample_label) in enumerate(self.dataset):
            if sum(sample_label) == 0:
                self.healthy_samples += [sample_data]
            else:
                self.sick_samples += [sample_data]

    def __getitem__(self, idx):
        random.seed(idx)
        sick_sample = self.sick_samples[random.randint(0, len(self.sick_samples) - 1)]
        healthy_sample = self.healthy_samples[random.randint(0, len(self.healthy_samples) - 1)]

        sick_idx = []
        sick_count = random.randint(1, self.slices_num//2+1)  # somewhere from 0-0.5 of slices should be sick
        for idx in range(sick_count):
            sick_idx.append(random.randint(0, self.slices_num-1))

        mix_sample = []
        for idx in range(self.slices_num):
            if idx in sick_idx:
                t = random.randint(0, self.slices_num-1)
                mix_sample.append(sick_sample[t])
            else:
                t = random.randint(0, self.slices_num-1)
                mix_sample.append(healthy_sample[t])

        mix_sample = torch.stack(mix_sample, dim=0)
        sick_idx = np.concatenate(sick_idx)
        return (mix_sample, sick_idx), self.label

    def __len__(self):
        return self.len

    def get_labels(self):
        return [self.label] * len(self.samples)

    def get_samples(self):
        raise NotImplementedError
