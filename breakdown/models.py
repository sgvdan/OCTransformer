import torch
import torch.nn as nn
import torch.nn.functional as F
import data


class ResBlock(nn.Module):
    def __init__(self, num, c_in):
        super().__init__()

        self.conv3d_block = \
            [nn.Conv3d(in_channels=c_in, out_channels=c_in, kernel_size=3, stride=1, padding=1, dilation=1,
                       groups=1) for _ in range(num)]

    def forward(self, x):
        x1 = x
        for l in self.conv3d_block:
            x1 = F.relu(l(x1))
        return x + x1


class RepresentationNet(nn.Module):
    def __init__(self, c_in):
        super().__init__()

        self.conv3d_1 = nn.Conv3d(in_channels=c_in, out_channels=8, kernel_size=3, stride=1, padding=0, dilation=1,
                                  groups=1)
        self.conv3d_2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0, dilation=1,
                                  groups=1)
        self.conv3d_3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, dilation=1,
                                  groups=1)
        self.conv3d_4 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=1,
                                  groups=1)
        self.conv3d_5 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1,
                                  groups=1)

        self.block1 = ResBlock(3, c_in)
        self.block2 = ResBlock(3, 8)
        self.block3 = ResBlock(3, 16)
        self.block4 = ResBlock(3, 32)
        self.block5 = ResBlock(3, 64)
        self.block6 = ResBlock(3, 128)

        self.pool = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.fc1 = nn.Linear(232960, 16384)
        self.fc2 = nn.Linear(16384, 4096)
        self.fc3 = nn.Linear(4096, 1024)

    def forward(self, x):
        # input size = (N,C_in,D,H,W)
        # print(x.shape)
        x = self.pool(F.relu(self.block1(x)))
        # print(x.shape)

        x = F.relu(self.conv3d_1(x))
        # print(x.shape)

        x = self.pool(F.relu(self.block2(x)))
        # print(x.shape)

        x = F.relu(self.conv3d_2(x))
        # print(x.shape)

        x = self.pool(F.relu(self.block3(x)))
        # print(x.shape)

        x = F.relu(self.conv3d_3(x))
        # print(x.shape)

        x = self.pool(F.relu(self.block4(x)))
        # print(x.shape)

        x = F.relu(self.conv3d_4(x))
        # print(x.shape)

        x = self.pool(F.relu(self.block5(x)))
        # print(x.shape)

        x = F.relu(self.conv3d_5(x))
        # print(x.shape)

        x = self.pool(F.relu(self.block6(x)))
        # print(x.shape)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    path = "C:/Users/guylu/Desktop/prev_files/Weizmann/OCT/test"
    d = data.OCT_Vol_DataSet(path)
