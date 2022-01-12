import math
import random
import torch
import torchvision.transforms as transforms


class dot_dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def make_deterministic(seed=0):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode=True)


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


def get_balance_weights(dataset, num_classes):
    labels = dataset.get_labels()
    num_scans = len(dataset)
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
