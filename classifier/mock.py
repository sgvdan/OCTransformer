import random
import torch

import util
from classifier.models.vit import MyViT

if __name__ == '__main__':
    util.make_deterministic()

    batch_size = 1
    height, width = (496, 1024)
    channels = 3
    slices = 37
    embedding_dim = 768
    x = torch.rand((batch_size, slices, channels, height, width))

    vit = MyViT(embedding_dim=embedding_dim)
    y = vit.forward(x)

    print(y)