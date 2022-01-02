from functools import partial

import torch
from timm.models.vision_transformer import VisionTransformer
from torchvision.models import resnet18


class MyViT(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()

        self.config = config
        backbone = partial(Backbone)
        self.model = VisionTransformer(img_size=(496, 512), patch_size=(16, 16),
                                       in_chans=3, num_classes=4,
                                       embed_dim=12, depth=12, num_heads=12, mlp_ratio=4.,
                                       qkv_bias=True,
                                       representation_size=None, distilled=False, drop_rate=0., attn_drop_rate=0.,
                                       drop_path_rate=0., embed_layer=backbone, norm_layer=None, act_layer=None,
                                       weight_init='')  # TODO: Migrate many of these to config.py

    def forward(self, x):
        return self.model(x)


class Backbone(torch.nn.Module):
    def __init__(self, embed_dim, num_patches=30, **kwargs):
        super().__init__()
        self.resnet = resnet18(pretrained=False, num_classes=embed_dim)  # TODO: pretrained on ImageNet?
        self.num_patches = 64 # TODO: Should be flexible # of slices per volume

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.reshape(batch_size , channels, height, width)
        x = self.resnet(x)
        x = x.reshape(batch_size, self.num_patches, -1)

        return x


class Resnet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet18(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.reshape(batch_size, channels, height, width)
        x = self.resnet(x)
        x = x.reshape(batch_size, -1)

        return x
