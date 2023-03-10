import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import symbols


def printArray(array):
    h, w = array.shape
    for i in range(h):
        for j in range(w):
            if array[i][j] > 0:
                print(255, end=' ')
            else:
                print(0, end=' ')
        print(end='\n')


def tresh(array):
    h, w = array.shape
    for i in range(h):
        for j in range(w):
            if array[i][j] <= 128:
                array[i][j] = 0
            else:
                array[i][j] = 255
    return array


def CustomTransform(image):
    kernel = np.array([[0, 0, 0],
                       [0, 1, 1],
                       [0, 1, 0]], np.uint8)
    image_dilate = cv2.dilate(image, kernel, iterations=1)

    return image_dilate


class CustomImageDataset(Dataset):

    def __init__(self, img_dir, annotations_file, transform=None,
                 target_transform=None):
        self.img_dir = os.path.join(img_dir, annotations_file)
        with open(os.path.join(img_dir, 'labels_{}.txt'.format(annotations_file))) as labels_file:
            l = labels_file.read().splitlines()
            self.img_labels = pd.DataFrame(l)

        self.transform = transform
        self.target_transform = target_transform
        # print(self.img_labels)

    def __len__(self):
        # return len(self.img_labels)
        return len(self.img_labels)*2

    def __getitem__(self, idx):
        parity = idx % 2
        idx = idx//2

        img_path = os.path.join(self.img_dir, '{}.jpg'.format(idx))
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = tresh(image)

        # cv2.imshow('Popravljena slika', image)
        # printArray(image)
        # print(image.shape)
        # cv2.imshow('Slika', image)
        # cv2.waitKey(0)

        if parity:
            image = CustomTransform(image)
            # cv2.imshow('Podebljana slika', image)
            # cv2.waitKey(0)

        label = self.img_labels.iloc[idx, 0]
        index = symbols.symbol2number(label)
        # OBRATI PAZNJU
        label = index

        if self.transform:
            image = self.transform(image)
            # print(image.size())

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


if __name__ == '__main__':

    dir = '/home/filip/Desktop/informatika/Petnica_project_2020-21/dataset'
    annotation_file = 'training'
    validation_data = CustomImageDataset(dir, annotation_file, ToTensor())

    validation_dataloader = DataLoader(
        validation_data, batch_size=10, shuffle=True)
    test_data = validation_data.__getitem__(1)
