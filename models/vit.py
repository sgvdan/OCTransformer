from functools import partial
import torch
import torchvision
from timm.models.vision_transformer import VisionTransformer

from mgu_models.nets.OSNet import OSMGUNet
from mgu_models.nets.TSNet import TSMGUNet
from util import move2cpu
from utils.vis import vis_result


class MyViT(torch.nn.Module):
    def __init__(self, backbone, config):
        super().__init__()

        self.config = config

        backbone = partial(BackboneWrapper, backbone=backbone, num_patches=self.config.num_slices)
        self.model = VisionTransformer(img_size=self.config.input_size, patch_size=(self.config.embedding_dim, 1),
                                       in_chans=3, num_classes=len(self.config.labels),
                                       embed_dim=self.config.embedding_dim, depth=self.config.vit_depth,
                                       num_heads=self.config.attention_heads, mlp_ratio=4., qkv_bias=True,
                                       representation_size=None, distilled=False, drop_rate=0., attn_drop_rate=0.,
                                       drop_path_rate=0., embed_layer=backbone, norm_layer=None, act_layer=None,
                                       weight_init='')  # TODO: Migrate many of these to config.py

    def forward(self, x):
        return self.model(x)

    def get_last_selfattention(self, x):
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.model.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)

        for i, blk in enumerate(self.model.blocks):
            if i < len(self.model.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                # return blk(x, return_attention=True)
                x = blk.norm1(x)
                B, N, C = x.shape
                qkv = blk.attn.qkv(x).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]  .to(self.config.device) # make torchscript happy (cannot use tensor as tuple)

                attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
                attn = attn.softmax(dim=-1)
                attn = blk.attn.attn_drop(attn)

                return attn

    def get_attention_map(self, volume):
        volume = volume.to(device=self.config.device, dtype=torch.float)

        attentions = self.get_last_selfattention(volume)

        nh = attentions.shape[1]  # number of heads

        # keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        # Average over all heads
        attentions = attentions.mean(dim=0, keepdim=True)

        # Perform temperature-Softmax to emphasize the relevant slices
        attentions = torch.softmax(attentions/self.config.attention_temperature, dim=1)

        return move2cpu(attentions)


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


class MGUNetBackboneWrapper(torch.nn.Module):
    def __init__(self, backbone, num_patches, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.num_patches = num_patches

        self.mgu_net = torch.nn.DataParallel(TSMGUNet()).cuda()
        checkpoint = torch.load('/home/projects/ronen/sgvdan/workspace/projects/OCTransformer/.models_bank/MGUNet_BOE_2015-pretrained/model_best.pth.tar')
        self.mgu_net.load_state_dict(checkpoint['state_dict'])

        # TODO: delete
        self.accum_resnet_tokens = torch.empty(0).to(device='cuda')
        self.accum_gist_tokens = torch.empty(0).to(device='cuda')
        self.accum_concat_tokens = torch.empty(0).to(device='cuda')

    def forward(self, x):
        x, gist = x
        batch_size, slices, channels, height, width = x.shape

        x = x.reshape(batch_size * slices, channels, height, width)
        y = self.backbone(x)
        y = y.reshape(batch_size, slices, -1)

        gist = gist.reshape(batch_size, slices, -1)

        x = torch.concat([y, gist], dim=-1)

        # TODO: delete
        self.accum_resnet_tokens = torch.concat([self.accum_resnet_tokens, y])
        self.accum_gist_tokens = torch.concat([self.accum_gist_tokens, gist])
        self.accum_concat_tokens = torch.concat([self.accum_concat_tokens, x])
        # TODO: delete

        return x
