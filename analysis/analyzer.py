import math
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance

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


def plot_gradient_heatmap(name, volume, label, model, optimizer, std_thresh=2, device="cuda"):
    n_slices = volume.shape[0]

    # Move to device
    original_volume = volume
    volume = volume.unsqueeze(0).to(device=device, dtype=torch.float)
    label = torch.clone(1 - label).detach().to(device=device, dtype=torch.long)
    volume.requires_grad = True
    # Run the model on the input image
    pred = model(volume)
    # Calculate the loss for this image
    loss = torch.nn.functional.cross_entropy(pred, label)
    # Backprop the gradients to the image
    optimizer.zero_grad()
    loss.backward()

    # gradients
    grad_map = volume.grad.squeeze().sum(axis=1)
    # threshold using std
    grad_map = (grad_map > (grad_map.mean() + (grad_map.std() * std_thresh))).float()

    # normalized numpy
    grad_map = np.uint8(cv2.normalize(tensor2im(grad_map), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))

    original_volume = move2cpu(original_volume.permute(0, 2, 3, 1))
    zeros = np.zeros_like(grad_map)
    # present grad mask as blue
    try:
        grad_mask = np.stack([grad_map.astype(np.int64), zeros, zeros], axis=3)
    except Exception as e:
        print('shit')

    masked = original_volume + grad_mask

    path = Path('./output/gradcum') / name
    os.makedirs(path, exist_ok=True)

    x_axis = np.arange(18-n_slices//2, 18 + n_slices//2+1, step=1)
    for i, slice in enumerate(masked):
        slice = (slice - masked.min()) / (masked.max()-masked.min()) * 255
        img = Image.fromarray(slice.astype(np.uint8)).convert('RGB')
        enhancer = ImageEnhance.Brightness(img)
        img_enhanced = enhancer.enhance(2.5)

        img_enhanced.save(path / 'slice_{}.png'.format(x_axis[i]))

    return grad_map
