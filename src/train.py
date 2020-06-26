from torchvision import datasets, transforms, utils
from torch import nn, optim, cuda, backends
from torch.autograd import Variable
from torch.utils import data
import modules
import numpy as np
import torch
import torch.nn.functional as F

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device', device)

    path = './Data'

    train_data = datasets.MNIST(root=path, train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root=path, train=False, download=True, transform=transforms.ToTensor())

    train_loader = data.DataLoader(train_data, batch_size=144, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = data.DataLoader(test_data, batch_size=144, shuffle=False, num_workers=1, pin_memory=True)

    net = modules.PixelCNN(in_channels=1, hidden_dims=64, out_channels=256, kernel_size=7, num_hidden_blocks=7)
    net = net.to(device)

    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, image in enumerate(train_loader, 0):
            # get the inputs; image is a list of [inputs, labels]
            inputs, labels = image
            inputs = inputs.to(device)
            labels = labels.to(device)

            target = Variable(inputs[:,0,:,:]*255).long()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

if __name__ == '__main__':
    main()