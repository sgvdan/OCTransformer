from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode

from data.hadassah_data import VolumeNormalize


class BOEChiuDataset(Dataset):
    def __init__(self, root, transformations=None):
        super().__init__()
        self.transformations = transformations
        self.root = Path(root)

        masks_path = self.root / 'mask'
        images_path = self.root / 'img'

        self.samples = {}
        for (image_path, mask_path) in zip(images_path.iterdir(), masks_path.iterdir()):
            subject = image_path.stem.split('-')[0]
            suffix = image_path.stem.split('-')[-1]

            if subject not in self.samples.keys():
                self.samples[subject] = []

            if 'flip' in suffix:  # ignore flips
                continue

            self.samples[subject].append((image_path, mask_path))

    def __getitem__(self, idx):
        subject = list(self.samples)[idx]

        images = []
        labels = []
        trans = transforms.Compose([transforms.ToTensor(), VolumeNormalize()])
        for image_path, label_path in self.samples[subject]:
            images.append(trans(Image.open(image_path)))
            labels.append(trans(Image.open(label_path))[0] * 255)

        images = torch.stack(images)
        labels = torch.stack(labels)

        intensities = [26, 51, 77, 102, 128, 153, 179, 204, 0, 255]
        for idx, intensity in enumerate(reversed(intensities)):
            labels[labels == intensity] = len(intensities) - 1 - idx  # so as to avoid setting 26->0->8

        if self.transformations is not None:
            images = self.transformations(images)

        labels = transforms.Resize((256, 256), InterpolationMode.NEAREST)(labels)

        return images, labels

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return torch.stack([label for _, label in self.samples])

    def get_classes(self):
        return None


def get_boe_chiu_transform(image_size):
    return transforms.Resize(image_size)


if __name__ == '__main__':
    a = BOEChiuDataset('/home/projects/ronen/sgvdan/workspace/datasets/2015_BOE_Chiu/mat_dataset/train')