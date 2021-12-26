from torchvision import transforms
from data.data import CachedDataset


class E2EVolumeDataset(CachedDataset):
    def __init__(self, cache, transformations=None):
        super().__init__(cache)
        self.transformations = transformations

    def __getitem__(self, idx):
        volume, *other = CachedDataset.__getitem__(self, idx)
        if self.transformations is not None:
            volume = self.transformations(volume)

        volume = volume.unsqueeze(dim=1).expand(-1, 3, -1, -1)
        return volume, *other


def get_hadassah_transform(image_size):
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Resize(image_size)])