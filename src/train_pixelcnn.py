from torchvision import datasets, transforms, utils
from torch import nn, optim, cuda, backends
from torch.autograd import Variable
from torch.utils import data
import modules
import numpy as np
import torch
import torch.nn.functional as F
import generate_pixelcnn

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device', device)

    path = './Data'
    batch_size = 64

    train_data = datasets.MNIST(root=path, train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root=path, train=False, download=True, transform=transforms.ToTensor())

    train_loader = data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = data.DataLoader(test_data, batch_size=64, shuffle=False, num_workers=1, pin_memory=True)

    net = modules.PixelCNN()
    net = net.to(device)
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(5):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0

        for i, image in enumerate(train_loader, 0):
            # get the inputs; image is a list of [inputs, labels]
            inputs, labels = image
            target = Variable(inputs[:,0,:,:] * 255).long()

            inputs = inputs.to(device)
            labels = labels.to(device)
            target = target.to(device)

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

        # if (epoch + 1) % 5 == 0:
        #     generate_pixelcnn.generate_bw(net, device=device, image_path='../output/sample_{}.png'.format(epoch + 1), batch_size=64)
        #     net.train()
        #     print('Sample generated at epoch {}, continuing', epoch)

    torch.save(net.state_dict(), '../output/model.pt')
    # generate_pixelcnn.generate_bw(net, device=device, image_path='../output/sample.png', batch_size=64)

if __name__ == '__main__':
    # For whatever reasons torch fails to run with cuda without a main() function
    main()