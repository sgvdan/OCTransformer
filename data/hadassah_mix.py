# A dataset of synthetica volumes - composed of mix of healthy/sick slices, to evaluate the attention mechanism

import torch
from torch.utils.data import Dataset


class MixedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.samples = []

        self.healthy_samples = []
        self.sick_samples = []

        self.generate_mixed_samples()
        self.label = torch.tensor(1)

    def generate_mixed_samples(self):
        for idx, (sample_data, sample_label) in enumerate(self.dataset):
            if sum(sample_label) < 2:
                self.healthy_samples += [sample_data]
            else:
                self.sick_samples += [sample_data]

        for idx in range(min(len(self.sick_samples), len(self.healthy_samples))):
            sick_sample = self.sick_samples[idx]
            healthy_sample = self.healthy_samples[idx]
            mix_sample = torch.stack([healthy_sample[0],
                                      healthy_sample[1],
                                      sick_sample[2],
                                      healthy_sample[2],
                                      healthy_sample[3]], dim=0)
            self.samples += [mix_sample]

    def __getitem__(self, idx):
        return self.samples[idx], self.label  # Assume it is sick

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return [self.label] * len(self.samples)

    def get_samples(self):
        return self.samples