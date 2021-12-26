from data.data import CachedDataset


class E2EVolumeDataset(CachedDataset):
    def __init__(self, cache, transformations=None):
        super().__init__(cache, transformations=transformations)

    def __getitem__(self, idx):
        volume, *other = CachedDataset.__getitem__(self, idx)
        volume = volume.unsqueeze(dim=1).expand(-1, 3, -1, -1)
        return volume, *other
