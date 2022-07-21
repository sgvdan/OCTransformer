import torch
from torch import nn

# Obtained from: https://github.com/manzilzaheer/DeepSets/blob/master/PointClouds/classifier.py#L58

class PermEqui1_mean(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        xm = x.mean(1, keepdim=True)
        x = self.Gamma(x-xm)
        return x


class PermEqui2_mean(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui2_mean, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        xm = x.mean(1, keepdim=True)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm
        return x


class DeepSet(nn.Module):
    def __init__(self, backbone, x_dim, d_dim, num_classes, dof=1):
        """

        :param backbone:
        :param x_dim: backbone's output dim
        :param d_dim: the intermediate dim
        :param num_classes: number of classes to classify for
        """
        super().__init__()

        self.backbone = backbone

        if dof == 1:
            print('DeepSet DOF 1', flush=True)
            perm_equi = PermEqui1_mean
        elif dof == 2:
            print('DeepSet DOF 2', flush=True)
            perm_equi = PermEqui2_mean
        else:
            raise NotImplementedError

        self.phi = nn.Sequential(
            perm_equi(x_dim, d_dim),
            nn.ELU(inplace=True),
            perm_equi(d_dim, d_dim),
            nn.ELU(inplace=True),
            perm_equi(d_dim, d_dim),
            nn.ELU(inplace=True),
        )

        self.ro = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(d_dim, d_dim),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(d_dim, num_classes),
        )

    # Taken from SliverNet
    def nonadaptiveconcatpool2d(self, x, k):
        # concatenating average and max pool, with kernel and stride the same
        ap = torch.nn.functional.avg_pool2d(x, kernel_size=k, stride=k)
        mp = torch.nn.functional.max_pool2d(x, kernel_size=k, stride=k)
        return torch.cat([mp, ap], 1)

    def forward(self, x):
        batch_size, slices_num, channels, height, width = x.shape
        x = x.view(batch_size * slices_num, channels, height, width)

        if x.shape[0] > 100:  # Cuda & ResNet are having trouble with long vectors, so split
            split = torch.split(x, 100)
            temp_features = []
            for chunk in split:
                temp_features.append(self.backbone(chunk))
            features = torch.cat(temp_features)
        else:
            features = self.backbone(x)  # B x M x h x w - B=batch size, M=#slices_per_volume, h=height, w=width

        kernel_size = (features.shape[-2], features.shape[-1])
        features = self.nonadaptiveconcatpool2d(features, kernel_size).view(batch_size, slices_num, -1)

        phi_output = self.phi(features)
        sum_output = phi_output.mean(1)
        ro_output = self.ro(sum_output)
        return ro_output
