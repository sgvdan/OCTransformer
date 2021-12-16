import torch
from classifier.models.vit import MyViT
import time

if __name__ == '__main__':

    batch_size = 1
    height, width = (496, 1024)
    channels = 3
    slices = 37
    embedding_dim = 768
    x = torch.rand((batch_size, slices, channels, height, width))

    mock = MyViT(embedding_dim=embedding_dim)
    start = time.time()
    y = mock.forward(x)
    end = time.time()
    print('time: {}'.format(end-start))

    print(y)