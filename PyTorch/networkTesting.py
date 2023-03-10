import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from loading import CustomImageDataset
from neuralNetwork import Net

net = Net()
net.load_state_dict(torch.load('net.pth'))
print("Ned loaded!\n")

batch_size = 4

#DATA TEST LOADING
dir = '/home/filip/Desktop/informatika/Petnica_project_2020-21/dataset'
testsetBAZA = CustomImageDataset(dir, 'testing', transform=ToTensor())
testloaderBAZA = DataLoader(testsetBAZA, batch_size=batch_size)

correct = 0
total = 0

#save memory with no_grad()
with torch.no_grad():
    for data in testloaderBAZA:
        images, labels = data
        
        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

print("Accuracy: {}%".format(100*correct/total))