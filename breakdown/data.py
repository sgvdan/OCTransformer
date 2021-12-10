import OCT_Align
import torch
import os
from torchvision import transforms as transforms
from pathlib import Path


class OCT_Vol_DataSet(torch.utils.data.Dataset):
    def __init__(self, path):
        # load your dataset (how every you want, this example has the dataset stored in a json file
        # path = "C:/Users/guylu/Desktop/prev_files/Weizmann/OCT/test/AIA 03346 OS 13.01.2020.E2E"
        self.dataset = []
        label = 0
        for epe_path in os.listdir(path):
            if epe_path.endswith(".E2E"):
                epe_path = path + "/" + epe_path
                pil_vol = OCT_Align.align_E2E_dir(epe_path, "")
                tensor_list = [transforms.ToTensor()(pil_pic)[0].unsqueeze(0) for pil_pic in pil_vol]
                Vol_stack_3D = torch.stack(tensor_list, dim=1)
                self.dataset.append((Vol_stack_3D, label))
        pass

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return sample

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    d = OCT_Vol_DataSet("C:/Users/guylu/Desktop/tests")
    pass
