import torch

# from functools import partial
# import torch.nn as nn
# import timm
# from timm.models.vision_transformer import VisionTransformer
# from timm.models.vision_transformer_hybrid import HybridEmbed
# from torchvision.models import resnet18

#
# class Mock(nn.Module):
#     def __init__(self):
#         super().__init__()
#         embedding_dim = 768
#
#         self.model = VisionTransformer(img_size=(37*496, 1024), patch_size=(embedding_dim, 1), in_chans=3, num_classes=2,
#                                        embed_dim=embedding_dim, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
#                                        representation_size=None, distilled=False, drop_rate=0., attn_drop_rate=0.,
#                                        drop_path_rate=0., embed_layer=Backbone, norm_layer=None, act_layer=None,
#                                        weight_init='')
#
#     def forward(self, x):
#         y = self.model(x)
#         return y
#
#
# class Backbone(nn.Module):
#     def __init__(self, img_size, patch_size, in_chans, embed_dim):
#         super().__init__()
#         self.resnet = resnet18(pretrained=False, num_classes=embed_dim)
#         self.num_patches = 37
#
#     def forward(self, x):
#         batch_size, slices, channels, height, width = x.shape
#
#         x = x.reshape(batch_size * slices, channels, height, width)
#         x = self.resnet(x)
#         x = x.reshape(batch_size, slices, -1)
#
#         return x
from classifier.models.vit import MyViT

if __name__ == '__main__':
    batch_size = 1
    height, width = (496, 1024)
    channels = 3
    slices = 37
    x = torch.rand((batch_size, slices, channels, height, width))
    # x = x.permute(0, 2, 1, 3, 4)
    # x = x.reshape(batch_size, channels, slices * height, width)  # have one big image

    mock = MyViT()
    y = mock.forward(x)
    print(y)