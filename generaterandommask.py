from __future__ import print_function, division
import os
import torch
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,datasets
import random
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class RandomMaskDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset,mask_pixels):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.base_dataset = dataset
        self.mask_pixels  = mask_pixels


    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
  

        X,y=self.base_dataset[idx]

        mask          = torch.zeros(X.shape[1]*X.shape[2])
        choice_list   = [i for i in range(len(mask))]
        choices       = random.choices(choice_list,k=self.mask_pixels)

        mask[choices] = 1
        mask          = mask.view(X.shape[1], X.shape[2])
        

        return X, y, mask
if __name__ == '__main__':
    classifier_data = datasets.CIFAR10(root="data",train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))
    masked_dataset  = RandomMaskDataset(classifier_data, 10)
    print(masked_dataset[0][2].shape)
    print(masked_dataset[0][1])
    print(masked_dataset[0][0].shape)
    plt.imshow(masked_dataset[0][2].squeeze().detach().numpy()*255, cmap="gray")
    plt.show()

