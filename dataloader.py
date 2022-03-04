from __future__ import print_function, division
import os
import torch
from pytorch_lightning import LightningDataModule
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
import random
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class RandomMaskDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset, mask_pixels):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.base_dataset = dataset
        self.mask_pixels  = int(mask_pixels)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        
        X, y = self.base_dataset[idx]

        mask          = torch.zeros(X.shape[1] * X.shape[2])
        choice_list   = [i for i in range(len(mask))]
        choices       = random.choices(choice_list, k = self.mask_pixels)

        mask[choices] = 1
        mask          = mask.view(X.shape[1], X.shape[2])
        

        return X, y, mask


class RandomMaskDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset, mask_pixels):
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
        
        X, y = self.base_dataset[idx]

        mask          = torch.zeros(X.shape[1] * X.shape[2])
        choice_list   = [i for i in range(len(mask))]
        choices       = random.choices(choice_list, k = self.mask_pixels)

        mask[choices] = 1
        mask          = mask.view(X.shape[1], X.shape[2])
        

        return X, y, mask

class CustomCIFAR10DataModule(LightningDataModule):
    def __init__(self, mask_pixels, batch_size=64):
        super().__init__()
        self.mask_pixels     = mask_pixels
        self.batch_size      = batch_size
        self.transform_train =  transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                ])

        self.transform_test  = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  ])

    def prepare_data(self):
        CIFAR10(os.getcwd(), train = True,  download = True, transform = self.transform_train)
        CIFAR10(os.getcwd(), train = False, download = True, transform = self.transform_test)

        self.transform_test   = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  ])

    def prepare_data(self):
        CIFAR10(os.getcwd(), train = True,  download = True, transform = self.transform_train)
        CIFAR10(os.getcwd(), train = False, download = True, transform = self.transform_test)

    def setup(self):
        transform      = transforms.Compose([transforms.ToTensor()])
        cifar10_train  = CIFAR10(os.getcwd(), train=True,  download=False, transform=self.transform_train)
        cifar10_test   = CIFAR10(os.getcwd(), train=False, download=False, transform=self.transform_test)
        custom_cifar10_train = RandomMaskDataset(cifar10_train, self.mask_pixels)

        self.train_dataset, self.val_dataset = random_split(custom_cifar10_train, [40000, 10000])
        self.test_dataset = cifar10_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def visualizer_dataloader(self):
        num_train_samples = 5
        sample_ds         = Subset(self.train_dataset, np.arange(num_train_samples))
        sample_sampler    = RandomSampler(sample_ds)
        sample_dl         = DataLoader(sample_ds, sampler=sample_sampler, batch_size=1)

        return sample_dl


if __name__ == '__main__':
    CIFAR10_dm = CustomCIFAR10DataModule()
    CIFAR10_dm.prepare_data()
    CIFAR10_dm.setup()
    td = CIFAR10_dm.train_dataloader()
    print(len(td))
