import math
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
from pytorch_grad_cam.utils.image import show_cam_on_image
from util import move2cpu, normalize


def plot_slices(volume, path):
    n_slices = volume.shape[0]

    x_axis = np.arange(18-n_slices//2, 18 + n_slices//2+1, step=1)

    for i in range(n_slices):
        image = move2cpu(volume[i])

        img = Image.fromarray(image[0, :, :]).convert('RGB')
        enhancer = ImageEnhance.Brightness(img)
        img_enhanced = enhancer.enhance(2.5)

        img_enhanced.save(path / 'slice-{}.png'.format(x_axis[i]))


def plot_attention(attention, path):
    n_heads = attention.shape[0]
    n_slices = attention.shape[1]

    x_axis = np.arange(18-n_slices//2, 18 + n_slices//2+1, step=1)

    plt.figure(figsize=(10, 10))
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


def plot_overlay_weighted_gradcam(volume, attention, cam, path, std_thresh=3,
                                  kernel_size=30, contour_threshold=0.5, smallest_contour_len=30, device='cuda'):
    original_images = move2cpu(normalize(volume.permute(0, 2, 3, 1)))
    n_slices = volume.shape[0]

    x_axis = np.arange(18-n_slices//2, 18 + n_slices//2+1, step=1)

    attention = move2cpu(attention)

    attention = np.exp(attention[0, :, None, None])
    weighted_cam = (attention / attention.max()) * cam

    threshold = (weighted_cam > std_thresh * weighted_cam.std())
    grad_map = threshold * weighted_cam
    #
    # grad_map = weighted_cam - weighted_cam.mean()
    # grad_map = (grad_map > std_thresh * grad_map.std()).astype(float)
    #
    # # blur
    # grad_map = torch.nn.functional.conv2d(input=grad_map.view(torch.Size([1, 1]) + grad_map.shape),
    #                                       weight=(torch.ones(1, 1, kernel_size,
    #                                                          kernel_size) / kernel_size ** 2).cuda(),
    #                                       padding='same').squeeze()
    # # normalized numpy
    # grad_map = np.uint8(cv2.normalize(tensor2im(grad_map), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
    # # contours
    # grad_mask = mask_from_heatmap(grad_map, thresh=contour_threshold, smallest_contour_len=smallest_contour_len)

    for i in range(weighted_cam.shape[0]):
        visualization = show_cam_on_image(original_images[i], grad_map[i], use_rgb=True)
        img = Image.fromarray(visualization)
        img.save(path / 'gradcam-{}.png'.format(x_axis[i]))