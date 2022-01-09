import torch
from torchvision import transforms
from data.data import CachedDataset


class E2EVolumeDataset(CachedDataset):
    def __init__(self, cache, samples_num, transformations=None):  # TODO: REMOVE samples_num
        super().__init__(cache)
        self.transformations = transformations
        self.samples_num = samples_num  # TODO: REMOVE

    def __getitem__(self, idx):
        volume, *other = CachedDataset.__getitem__(self, idx)
        if self.transformations is not None:
            volume = self.transformations(volume)

        indices = torch.tensor(range(18-self.samples_num//2, 18 + self.samples_num//2+1))  # TODO: REMOVE
        volume = volume.index_select(dim=0, index=indices)  # TODO: REMOVE

        volume = volume.unsqueeze(dim=1).expand(-1, 3, -1, -1)
        return volume, *other


def get_hadassah_transform(image_size):
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Resize(image_size)])
