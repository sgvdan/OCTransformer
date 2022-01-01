from functools import partial

import torch
from timm.models.vision_transformer import VisionTransformer
from torchvision.models import resnet18


class MyViT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        backbone = partial(Backbone, num_patches=self.config.num_slices)
        self.model = VisionTransformer(img_size=self.config.input_size, patch_size=(self.config.embedding_dim, 1),
                                       in_chans=3, num_classes=self.config.num_classes,
                                       # TODO: CHANGE NUM_CLASSES ACCORDING TO DATASET
                                       embed_dim=self.config.embedding_dim, depth=12, num_heads=12, mlp_ratio=4.,
                                       qkv_bias=True,
                                       representation_size=None, distilled=False, drop_rate=0., attn_drop_rate=0.,
                                       drop_path_rate=0., embed_layer=backbone, norm_layer=None, act_layer=None,
                                       weight_init='')  # TODO: Migrate many of these to config.py

    def forward(self, x):
        return self.model(x)


class Backbone(torch.nn.Module):
    def __init__(self, embed_dim, num_patches, **kwargs):
        super().__init__()
        self.resnet = resnet18(pretrained=False, num_classes=embed_dim)  # TODO: pretrained on ImageNet?
        self.num_patches = num_patches  # TODO: Should be flexible # of slices per volume

    def forward(self, x):
        batch_size, slices, channels, height, width = x.shape

        x = x.reshape(batch_size * slices, channels, height, width)
        x = self.resnet(x)
        x = x.reshape(batch_size, slices, -1)

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
