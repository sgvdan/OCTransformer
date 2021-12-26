from torchvision import transforms
from data.data import CachedDataset


class KermanyDataset(CachedDataset):
    def __init__(self, cache, transformations=None):
        super().__init__(cache)
        self.transformations = transformations

    def __getitem__(self, idx):
        image, *other = CachedDataset.__getitem__(self, idx)
        if self.transformations is not None:
            image = self.transformations(image)

        image = image.unsqueeze(dim=1).expand(-1, 3, -1, -1)
        return image, *other


def get_kermany_transform(image_size):
    return transforms.Resize(image_size)
