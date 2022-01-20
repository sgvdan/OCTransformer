from functools import partial

import torch
from timm.models.vision_transformer import VisionTransformer
from torchvision.models import resnet18


class MyViT(torch.nn.Module):
    def __init__(self, backbone, config):
        super().__init__()

        self.config = config
        backbone = partial(BackboneWrapper, backbone=backbone, num_patches=self.config.num_slices)
        self.model = VisionTransformer(img_size=self.config.input_size, patch_size=(self.config.embedding_dim, 1),
                                       in_chans=3, num_classes=self.config.num_classes,
                                       embed_dim=self.config.embedding_dim, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                                       representation_size=None, distilled=False, drop_rate=0., attn_drop_rate=0.,
                                       drop_path_rate=0., embed_layer=backbone, norm_layer=None, act_layer=None,
                                       weight_init='')  # TODO: Migrate many of these to config.py

    def forward(self, x):
        return self.model(x)


class BackboneWrapper(torch.nn.Module):
    def __init__(self, backbone, num_patches, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.num_patches = num_patches

    def forward(self, x):
        batch_size, slices, channels, height, width = x.shape

        x = x.reshape(batch_size * slices, channels, height, width)
        x = self.backbone(x)
        x = x.reshape(batch_size, slices, -1)

        return x
