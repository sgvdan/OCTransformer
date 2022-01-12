from torch.utils.data import Dataset
from torchvision import transforms


class KermanyDataset(Dataset):
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
