import math
import random
import numpy as np
import torch
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

# def mask_from_heatmap(image, thresh=0.9, smallest_contour_len=30):
#     # binary mask according to threshold
#     ret, thresh_img = cv2.threshold(src=image, thresh=thresh * 255, maxval=255, type=cv2.THRESH_BINARY)
#     # find contours
#     contours, hierarchy = cv2.findContours(image=thresh_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
#     # filter out irrelevant contours, and smooth the rest
#     contours = filter_contours(contours, smallest_contour_len=smallest_contour_len)
#     try:
#         contours = smoothene_contours(contours=contours)
#     except:
#         print("Failed to smoothen contours :/    continuing...")
#     # draw only filled contours
#     mask = np.zeros(image.shape)
#     cv2.fillPoly(img=mask, pts=contours, color=1)
#     return mask
