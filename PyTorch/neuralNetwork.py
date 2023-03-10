from loading import CustomImageDataset
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def printArray(array):
    h, w = array.shape
    for i in range(h):
        for j in range(w):
            if array[i][j] > 0:
                print(255, end=' ')
            else:
                print(0, end=' ')
        print(end='\n')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        #in_channels, out_channels, kernel_size
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, 5)

        #in_features, out_features
        # 16*5*5-CIFAR10__________16*8*8-BAZA
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(40 * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        # 84,82 je izlaz za CIFAR10, dok je 84,82 izlaz za BAZU
        #self.fc3 = nn.Linear(84, 10)
        self.fc3 = nn.Linear(1000, 82)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        #torch.flatten(input, start_dim=0, end_dim=-1)
        # -1 znaci da komp odredi tu dimenziju
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
        # return F.softmax(x, dim=0)


if __name__ == '__main__':

    net = Net()

    # DATA TRAINING LOADING
    from loading import CustomImageDataset

    batch_size = 4

    dir = '/home/filip/Desktop/informatika/Petnica_project_2020-21/dataset'

    trainsetBAZA = CustomImageDataset(dir, 'training', transform=ToTensor())
    trainloaderBAZA = DataLoader(trainsetBAZA, batch_size=batch_size,
                                 shuffle=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # Traing the network
    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(trainloaderBAZA, 0):

            inputs, labels = data
            # print(labels)

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print("Training finished!\n")

    # SAVING MODEL
    torch.save(net.state_dict(), "net.pth")
    print("Saved PyTorch Model into net.pth")

    # LOADING MODEL
    # net.load_state_dict(torch.load('net.pth'))
    # print("Ned loaded!\n")

    # #DATA TEST LOADING
    # testsetBAZA = CustomImageDataset(dir, 'test', transform=ToTensor())
    # testloaderBAZA = DataLoader(testsetBAZA, batch_size=batch_size,
    #                                 shuffle=True, num_workers=2)

    # correct = 0
    # total = 0

    # #save memory with no_grad()
    # with torch.no_grad():
    #     for data in testloaderBAZA:
    #         images, labels = data

    #         outputs = net(images)

    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted==labels).sum().item()

    # print("Accuracy: {}%".format(100*correct/total))
