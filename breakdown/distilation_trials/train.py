import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import kermany_dataset
import kermany_net
from config import default_config
import utils

if __name__ == '__main__':

    batch_size = 4
    epochs = 2
    trainset = kermany_dataset.Kermany_DataSet("")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    valset = kermany_dataset.Kermany_DataSet("C:/Users/guylu/Desktop/tests")
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    config = 0
    model = kermany_net.MyViT(config)
    criterion = nn.CrossEntropyLoss #nn.L1Loss
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
