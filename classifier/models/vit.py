import torch
from timm.models.vision_transformer import VisionTransformer
from torchvision.models import resnet18


class MyViT(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.model = VisionTransformer(img_size=(37*496, 1024), patch_size=(embedding_dim, 1), in_chans=3, num_classes=2,
                                       embed_dim=embedding_dim, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                                       representation_size=None, distilled=False, drop_rate=0., attn_drop_rate=0.,
                                       drop_path_rate=0., embed_layer=Backbone, norm_layer=None, act_layer=None,
                                       weight_init='')

    def forward(self, x):
        y = self.model(x)
        return y


class Backbone(torch.nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.resnet = resnet18(pretrained=False, num_classes=embed_dim)
        self.num_patches = 37

    def forward(self, x):
        batch_size, slices, channels, height, width = x.shape

        x = x.reshape(batch_size * slices, channels, height, width)
        x = self.resnet(x)
        x = x.reshape(batch_size, slices, -1)

        return x
