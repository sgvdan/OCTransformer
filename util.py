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


def get_balance_class_weights(labels):
    pos = labels.sum(dim=0)
    neg = (1 - labels).sum(dim=0)
    weights = neg / pos

    assert not torch.any(weights.isnan()) and not torch.any(weights.isinf())
    print('Labels prevalence are:', pos, 'corresponding class-balancing weights used:', weights)  # Log important info

    return weights


def move2cpu(d):
    """Move data from gpu to cpu"""
    return d.detach().cpu().float().numpy()


def normalize(d):
    d = d.float()
    d -= d.min()
    d /= d.max()
    return d


def max_contour(data):
    xs, ys = [], []
    for x in np.unique(data[:, 0]):
        mask = data[:, 0] == x  # gather all similar x
        xs.append(x)
        ys.append(np.max(data[mask, 1]))  # y = max. y

    return xs, ys

def figure2img(plt):
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    return Image.open(buffer)


def get_reduced_label(label, keys):
    return torch.tensor([label[key] for key in keys])
