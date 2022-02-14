from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad

import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
from pytorch_grad_cam.utils.image import show_cam_on_image

from util import move2cpu, tensor2im


def plot_attention(name, volume, attention):
    n_heads = attention.shape[0]
    n_slices = attention.shape[1]

    path = Path('./output/attention') / name
    os.makedirs(path, exist_ok=True)

    x_axis = np.arange(18-n_slices//2, 18 + n_slices//2+1, step=1)

    for i in range(n_slices):
        image = move2cpu(volume[i])

        img = Image.fromarray(image[0, :, :]).convert('RGB')
        enhancer = ImageEnhance.Brightness(img)
        img_enhanced = enhancer.enhance(2.5)

        img_enhanced.save(path / 'slice_{}.png'.format(x_axis[i]))

    plt.figure(figsize=(10, 10))
    rows_num = round(math.sqrt(n_heads))
    cols_num = -(n_heads // -rows_num)

    for i in range(n_heads):
        axes = plt.subplot(rows_num, cols_num, i+1)
        attention = move2cpu(attention.unsqueeze(dim=0))

        norm_image = (attention[i] - attention.min()) / (attention.max()-attention.min()) * 255
        plt.imshow(norm_image, cmap='gist_heat')
        plt.title(f"Head n: {i+1}")
        axes.set_xticks(list(range(n_slices)))
        axes.set_xticklabels([x_axis[i] for i in range(n_slices)])

    plt.yticks([])

    plt.tight_layout()
    plt.savefig(path / 'attention.png')


def grad_cam_model(res_model, input_tensor, label=0, device='cuda'):
    input_tensor = input_tensor.to(device=device, dtype=torch.float)
    original_image = input_tensor[0].permute(1,2,0).cpu().numpy()
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    res_model.eval()
    target_layers = [res_model.layer4[-1]]
    out = res_model(input_tensor)
    cams = [(GradCAM, 'GradCAM'), (ScoreCAM, 'ScoreCAM'), (GradCAMPlusPlus, 'GradCAMPlusPlus'),
            (AblationCAM, 'AblationCAM'), (XGradCAM, 'XGradCAM')]  #  (EigenCAM, 'EigenCAM'), (FullGrad, 'FullGrad')
    for cam_algo, cam_name in cams:
        cam = cam_algo(model=res_model, target_layers=target_layers,
                       use_cuda=True if torch.cuda.is_available() else False)
        target_category = label
        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
        img = Image.fromarray(visualization)
        img.save('temp/{}.png'.format(cam_name))

    return out


def plot_gradient_heatmap(name, volume, label, model, optimizer, std_thresh=2, device="cuda"):
    n_slices = volume.shape[0]

    # Move to device
    original_volume = volume
    # volume = volume.unsqueeze(0).to(device=device, dtype=torch.float)
    # label = torch.clone(1 - label).detach().to(device=device, dtype=torch.long)

    # TEMPORARY
    volume = volume.to(device=device, dtype=torch.float)
    label = torch.tensor([0] * 35).to(device=device, dtype=torch.uint8)
    # TEMPORARY

    volume.requires_grad = True
    # Run the model on the input image
    pred = model(volume)
    _, pred_types = pred.max(dim=1)
    # Calculate the loss for this image
    loss = torch.nn.functional.cross_entropy(pred, label)
    # Backprop the gradients to the image
    optimizer.zero_grad()
    loss.backward()

    # gradients
    grad_map = volume.grad.squeeze().sum(axis=1)
    # threshold using std
    grad_map = (grad_map - grad_map.mean() > std_thresh * grad_map.std()).float()

    # normalized numpy
    grad_map = np.uint8(cv2.normalize(tensor2im(grad_map), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))

    original_volume = move2cpu(original_volume.permute(0, 2, 3, 1))
    zeros = np.zeros_like(grad_map)
    # present grad mask as red
    grad_mask = np.stack([grad_map.astype(np.uint8), zeros, zeros], axis=3)

    masked = original_volume + grad_mask
    masked = ((masked - masked.min()) / (masked.max() - masked.min()) * 255).astype(np.uint8)

    path = Path('./output/gradcam') / name
    os.makedirs(path, exist_ok=True)

    # x_axis = np.arange(18-n_slices//2, 18 + n_slices//2+1, step=1)
    for i, slice in enumerate(masked):
        img = Image.fromarray(slice).convert('RGB')
        enhancer = ImageEnhance.Brightness(img)
        img_enhanced = enhancer.enhance(2.5)
        img_enhanced.save(path / 'slice_{}.png'.format(i))

    return grad_map
