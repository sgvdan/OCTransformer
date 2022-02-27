import math

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
from pytorch_grad_cam.utils.image import show_cam_on_image

from util import move2cpu, normalize


def plot_slices(volume, path, logger):
    n_slices = volume.shape[0]

    x_axis = np.arange(18-n_slices//2, 18 + n_slices//2+1, step=1)

    for i in range(n_slices):
        name = 'slice-{}.png'.format(x_axis[i])
        image = move2cpu(volume[i])

        img = Image.fromarray(image[0, :, :]).convert('RGB')
        enhancer = ImageEnhance.Brightness(img)
        img_enhanced = enhancer.enhance(2.5)

        img_enhanced.save(path / name)
        logger.log_image(img, caption=name)


def plot_attention(attention, path, logger):
    n_heads = attention.shape[0]
    n_slices = attention.shape[1]

    x_axis = np.arange(18-n_slices//2, 18 + n_slices//2+1, step=1)

    fig = plt.figure(figsize=(10, 10))
    rows_num = round(math.sqrt(n_heads))
    cols_num = -(n_heads // -rows_num)

    for i in range(n_heads):
        axes = plt.subplot(rows_num, cols_num, i+1)
        attention = move2cpu(attention.unsqueeze(dim=0))

        # norm_image = (attention[i] - attention.min()) / (attention.max()-attention.min()) * 255
        norm_image = (attention[i] - attention[i].min())
        plt.imshow(norm_image, cmap='gist_heat', norm=None)
        plt.title("Attention Map")
        axes.set_xticks(list(range(n_slices)))
        axes.set_xticklabels([x_axis[i] for i in range(n_slices)])

    plt.yticks([])
    plt.tight_layout()
    plt.savefig(path / 'attention.png')

    img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    logger.log_image(img, caption='attention')


def plot_gradcam(volume, cam, path, logger):
    n_slices = volume.shape[0]
    original_images = move2cpu(normalize(volume.permute(0, 2, 3, 1)))

    x_axis = np.arange(18-n_slices//2, 18 + n_slices//2+1, step=1)

    for i in range(volume.shape[0]):
        name = 'gradcam-{}.png'.format(x_axis[i])
        visualization = show_cam_on_image(original_images[i], 0.7*cam[i], use_rgb=True, colormap=cv.COLORMAP_HOT)
        img = Image.fromarray(visualization)
        logger.log_image(img, caption=name)
        img.save(path / name)


def plot_masks(volume, attention, cam, path, logger, std_thresh=3):
    n_slices = volume.shape[0]
    original_images = move2cpu(normalize(volume.permute(0, 2, 3, 1)))

    x_axis = np.arange(18-n_slices//2, 18 + n_slices//2+1, step=1)

    attention = move2cpu(attention)
    weighted_cam = attention[0, :, None, None] * cam

    threshold = (weighted_cam > std_thresh * weighted_cam.std())
    grad_map = threshold * (weighted_cam / attention.max())

    for i in range(weighted_cam.shape[0]):
        name = 'weighted-gradcam-{}.png'.format(x_axis[i])
        visualization = show_cam_on_image(original_images[i], 0.3*grad_map[i], use_rgb=True, colormap=cv.COLORMAP_HOT)
        img = Image.fromarray(visualization)
        logger.log_image(img, caption=name)
        img.save(path / name)
