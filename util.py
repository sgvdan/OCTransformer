import math
import random
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from cv2 import cv2

pseudo_random = random.Random()  # performed prior to make_deterministic


class dot_dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_pseudo_random():
    return pseudo_random


def make_deterministic(seed=0):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode=True)
    random.seed(seed)
    np.random.seed(seed)


def split_list(input_list, chunks, random_split=True):
    assert sum(chunks) == 1
    breakdown = []

    if random_split:
        random.shuffle(input_list)

    tail = 0
    accum = 0
    for chunk in chunks:
        accum += chunk
        head = math.floor(accum * len(input_list))
        breakdown.append(input_list[tail:head])

        tail = head

    return breakdown


def get_balance_weights(labels, num_classes):
    num_scans = len(labels)
    assert num_scans > 0 and num_classes > 0

    # Count # of appearances per each class
    count = [0] * num_classes
    for label in labels:  # TODO: If one_hot handling required - take `label.argmax(-1)`
        count[int(label)] += 1

    # Each class receives weight in reverse proportion to its # of appearances
    weight_per_class = [0.] * num_classes
    for idx in range(num_classes):
        weight_per_class[idx] = float(num_scans) / float(count[idx])

    # Assign class-corresponding weight for each element
    weights = [0] * num_scans
    for idx, label in enumerate(labels):
        weights[idx] = weight_per_class[int(label)]

    return torch.FloatTensor(weights)


def move2cpu(d):
    """Move data from gpu to cpu"""
    return d.detach().cpu().float().numpy()


def normalize(d):
    d -= d.min()
    d /= d.max()
    return d


def show_cams_on_image_batch(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmaps = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmaps = cv2.cvtColor(heatmaps, cv2.COLOR_BGR2RGB)
    heatmaps = np.float32(heatmaps) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cams = heatmaps + imgs
    cams = cams / np.max(cams)
    return np.uint8(255 * cams)


def figure2img(plt):
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    return Image.open(buffer)
