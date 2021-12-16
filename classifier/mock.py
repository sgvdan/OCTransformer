from functools import partial

import torch.nn as nn
import torch
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer_hybrid import HybridEmbed
from torchvision.models import resnet18

class Mock(nn.Module):
    def __init__(self):
        super().__init__()
        embedding_dim = 768

        self.backbone = Backbone(img_size=(496, 1024), embedding_dim=embedding_dim)
        embed_layer = partial(MyHybridEmbed, backbone=self.backbone)
        self.model = VisionTransformer(img_size=(37*496, 1024), patch_size=(embedding_dim, 1), in_chans=3, num_classes=2,
                                       embed_dim=embedding_dim, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                                       representation_size=None, distilled=False, drop_rate=0., attn_drop_rate=0.,
                                       drop_path_rate=0., embed_layer=embed_layer, norm_layer=None, act_layer=None,
                                       weight_init='')

    def forward(self, x):
        y = self.model(x)  # TODO: NEED TO MAKE THIS EXECUTE SO I CAN CLASSIFY!!


class Backbone(nn.Module):
    def __init__(self, img_size, embedding_dim):
        super().__init__()
        self.img_height, self.img_width = img_size
        self.resnet = resnet18(pretrained=False, num_classes=embedding_dim)

    def forward(self, x):
        batch_size, channels, overall_height, width = x.shape
        slices = overall_height // self.img_height

        x = x.reshape(batch_size, channels, slices, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch_size * slices, channels, height, width)
        x = self.resnet(x)
        x = x.reshape(batch_size, slices, -1)
        return x


class MyHybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=(224, 224), patch_size=(1, 1), feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        # self.img_size = img_size
        # self.patch_size = patch_size
        self.backbone = backbone
        self.num_patches = 37
        # if feature_size is None:
        #     with torch.no_grad():
        #         # NOTE Most reliable way of determining output dims is to run forward pass
        #         training = backbone.training
        #         if training:
        #             backbone.eval()
        #         o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
        #         if isinstance(o, (list, tuple)):
        #             o = o[-1]  # last feature if backbone outputs list/tuple of features
        #         feature_size = o.shape[-2:]
        #         feature_dim = o.shape[1]
        #         backbone.train(training)
        # else:
        #     feature_size = to_2tuple(feature_size)
        #     if hasattr(self.backbone, 'feature_info'):
        #         feature_dim = self.backbone.feature_info.channels()[-1]
        #     else:
        #         feature_dim = self.backbone.num_features
        # assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        # self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        # self.num_patches = self.grid_size[0] * self.grid_size[1]
        # self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.backbone(x)
        # if isinstance(x, (list, tuple)):
        #     x = x[-1]  # last feature if backbone outputs list/tuple of features
        # x = self.proj(x).flatten(2).transpose(1, 2)
        return x

if __name__ == '__main__':
    batch_size = 1
    height, width = (496, 1024)
    channels = 3
    slices = 37
    x = torch.rand((batch_size, slices, channels, height, width))
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(batch_size, channels, slices * height, width)  # have one big image

    mock = Mock()
    y = mock.forward(x)
    print(y)