import torch
import torch.nn as nn
import torch.nn.functional as F
import data


class Net(nn.Module):
    def __init__(self, c_in):
        super().__init__()

        self.conv3d_1 = nn.Conv3d(in_channels=c_in, out_channels=8, kernel_size=3, stride=1, padding=0, dilation=1,
                                  groups=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # input size = (N,C_in,D,H,W)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    path = "C:/Users/guylu/Desktop/prev_files/Weizmann/OCT/test/AIA 03346 OS 13.01.2020.E2E"
    d = data.OCT_Vol_DataSet(path)
