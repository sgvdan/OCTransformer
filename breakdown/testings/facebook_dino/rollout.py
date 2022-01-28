import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2
import wandb
from data_for_tests_FB import Kermany_DataSet


class dot_dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def grad_rollout2(attentions, gradients, discard_ratio=0.9):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):
            weights = grad
            attention_heads_fused = (attention * weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            # indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)
            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1:]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(62, 64).numpy()
    mask = mask / np.max(mask)
    return mask


class VITAttentionGradRollout:
    def __init__(self, model, attention_layer_name='attn_drop', discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
                module.register_backward_hook(self.get_attention_gradient)

        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())

    def __call__(self, input_tensor, category_index):
        self.model.zero_grad()
        output = self.model(input_tensor)
        del input_tensor
        category_mask = torch.zeros(output.size()).to(device)
        category_mask[:, category_index] = 1
        loss = (output * category_mask).sum()
        loss.backward()

        g = grad_rollout2(self.attentions, self.attention_gradients, self.discard_ratio)
        return g


# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap * 0.4 + np.float32(img)
    cam = cam / np.max(cam)
    return cam


if __name__ == '__main__':
    image = Image.open("../../../../data/kermany/val/DME/DME-9603124-1.jpeg")
    # image = Image.open("../../../../data/kermany/test/DME/DME-11053-1.jpeg")
    t = transforms.Compose([transforms.ToTensor()])
    im = t(image)
    im = torch.cat([im, im, im], 0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
    model.to(device)
    model.eval()
    print(im.shape)
    for i in [0.9, 0.8]:
        grad_rollout = VITAttentionGradRollout(model, discard_ratio=i, )
        mask = grad_rollout(im.unsqueeze(dim=0).to(device), category_index=2)
        plt.imshow(mask)
        plt.savefig(f"dme_fusion_{i}.png")
