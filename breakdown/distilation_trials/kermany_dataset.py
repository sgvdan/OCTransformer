import torch
import os
from torchvision import transforms as transforms
from pathlib import Path
from torchvision.io import read_image


class Kermany_DataSet(torch.utils.data.Dataset):
    def __init__(self, path):
        # load your dataset (how every you want, this example has the dataset stored in a json file
        # path = "C:/Users/guylu/Desktop/prev_files/Weizmann/OCT/test/AIA 03346 OS 13.01.2020.E2E"
        t = transforms.Compose([transforms.ToTensor])
        self.dataset = []
        label = 0
        label_dict = {"NORMAL": 0,
                      "CNV": 1,
                      "DME": 2,
                      "DRUSEN": 3}
        f_1 = lambda x: 0 if "NORMAL" in x else 0
        f_2 = lambda x: 1 if "CNV" in x else 0
        f_3 = lambda x: 2 if "DME" in x else 0
        f_4 = lambda x: 3 if "DRUSEN" in x else 0
        for path in Path(path).rglob('*.jpeg'):
            path = str(path)
            image = t(read_image(path))
            label = f_1(path) + f_2(path) + f_3(path) + f_4(path)
            self.dataset.append((image, label))


    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return sample

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    d = Kermany_DataSet("C:/Users/guylu/Desktop/tests")
    pass
