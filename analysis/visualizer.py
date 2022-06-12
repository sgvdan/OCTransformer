import math

import cv2 as cv
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
from pytorch_grad_cam.utils.image import show_cam_on_image

from sklearn.manifold import TSNE

from util import move2cpu, normalize, figure2img
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad


def get_gradcam(input_tensor, model, target_layers, targets, type, aug_smooth, eigen_smooth, device):
    input_tensor = input_tensor.to(device=device, dtype=torch.float)

    type = type.lower()
    if type == 'gradcam':
        cam = GradCAM
    elif type == 'scorecam':
        cam = ScoreCAM
    elif type == 'gradcam++':
        cam = GradCAMPlusPlus
    elif type == 'ablationcam':
        cam = AblationCAM
    elif type == 'xgradcam':
        cam = XGradCAM
    elif type == 'eigencam':
        cam = EigenCAM
    elif type =='fullgrad':
        cam = FullGrad
    else:
        raise NotImplementedError

    cam_instance = cam(model=model, target_layers=target_layers, use_cuda=(device == 'cuda'))
    return cam_instance(input_tensor=input_tensor, targets=targets, aug_smooth=aug_smooth, eigen_smooth=eigen_smooth)


def get_masks(attention, cam, std_thresh=3):
    weighted_cam = attention[0, :, None, None] * cam
    threshold = (weighted_cam > std_thresh * weighted_cam.std())
    grad_map = threshold * (weighted_cam / attention.max())

    return grad_map


def plot_slices(volume, logger, title):
    n_slices = volume.shape[0]

    x_axis = np.arange(18-n_slices//2, 18 + n_slices//2+1, step=1)

    for i in range(n_slices):
        img = move2cpu(volume[i])

        img = Image.fromarray(img[0, :, :] * 255).convert('RGB')
        enhancer = ImageEnhance.Brightness(img)
        img_enhanced = enhancer.enhance(2.5)

        logger.log_image(img_enhanced, caption=title + '-' + str(x_axis[i]))


def plot_attention(attention, logger, title):
    n_heads = attention.shape[0]
    n_slices = attention.shape[1]

    x_axis = np.arange(18-n_slices//2, 18 + n_slices//2+1, step=1)

    fig = plt.figure(figsize=(10, 10))
    rows_num = round(math.sqrt(n_heads))
    cols_num = -(n_heads // -rows_num)

    for i in range(n_heads):
        axes = plt.subplot(rows_num, cols_num, i+1)
        attn = np.concatenate([np.array([0, 1]), attention[i]]).reshape(1, -1)
        plt.imshow(attn, cmap='gist_heat', norm=None)
        plt.title("Attention Map")
        axes.set_xticks(list(range(n_slices+2)))
        axes.set_xticklabels(['Min', 'Max'] + [x_axis[i] for i in range(n_slices)])

    plt.yticks([])
    plt.tight_layout()

    logger.log_image(figure2img(plt), caption=title)


def plot_gradcam(volume, cam, logger, title):
    n_slices = volume.shape[0]
    original_images = move2cpu(normalize(volume.permute(0, 2, 3, 1)))

    x_axis = np.arange(18-n_slices//2, 18 + n_slices//2+1, step=1)

    for i in range(volume.shape[0]):
        visualization = show_cam_on_image(original_images[i], 0.7*cam[i], use_rgb=True, colormap=cv.COLORMAP_HOT)
        logger.log_image(Image.fromarray(visualization), caption=(title + '-' + str(x_axis[i])))


def plot_masks(volume, mask, logger, title):
    n_slices = volume.shape[0]
    original_images = move2cpu(normalize(volume.permute(0, 2, 3, 1)))

    x_axis = np.arange(18-n_slices//2, 18 + n_slices//2+1, step=1)

    for i in range(mask.shape[0]):
        visualization = show_cam_on_image(original_images[i], 0.3*mask[i], use_rgb=True, colormap=cv.COLORMAP_HOT)
        img = Image.fromarray(visualization)
        logger.log_image(img, caption=(title + '-' + str(x_axis[i])))


def low_dimension_plot(data, labels, title):
    import pandas as pd
    import seaborn as sns

    tsne = TSNE()
    z = tsne.fit_transform(data.cpu().numpy())

    df = pd.DataFrame()
    df["y"] = labels.cpu().numpy()
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 4),
                    data=df).set(title=title)
