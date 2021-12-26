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
    random.seed(seed)
    torch.use_deterministic_algorithms(mode=True)
