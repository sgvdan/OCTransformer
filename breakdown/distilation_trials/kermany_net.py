from functools import partial

import torch
from timm.models.vision_transformer import VisionTransformer
from torchvision.models import resnet18


class MyViT(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()

        self.config = config
        backbone = partial(Backbone)
        self.model = VisionTransformer(img_size=(496, 512), patch_size=config.vit_patch_size,
                                       in_chans=3, num_classes=config.vit_num_classes,
                                       embed_dim=config.vit_embed_dim, depth=config.vit_depth,
                                       num_heads=config.vit_num_heads, mlp_ratio=config.vit_mlp_ratio,
                                       qkv_bias=True, representation_size=None, distilled=False,
                                       drop_rate=config.vit_drop_rate, attn_drop_rate=config.vit_attn_drop_rate,
                                       drop_path_rate=0., embed_layer=backbone, norm_layer=None, act_layer=None,
                                       weight_init='')

    def forward(self, x):
        return self.model(x)


class Backbone(torch.nn.Module):
    def __init__(self, embed_dim, num_patches=30, **kwargs):
        super().__init__()
        self.num_patches = 64
        #self.config.backbone_pretrained
        self.resnet = resnet18(num_classes=embed_dim * self.num_patches, pretrained=False)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.reshape(batch_size, channels, height, width)
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
