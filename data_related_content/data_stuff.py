## pythonic imports
import random

## Torch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MaskedDataset(Dataset):
    def __init__(
        self,
        dataset,
        mask_sparsity,
        transforms,
        im_size,
        input_mask=None,
        mask_mode=True,
    ):
        self.dataset = dataset
        self.mask_sparsity = mask_sparsity
        self.transforms = transforms
        self.im_size = im_size
        self.input_mask = input_mask
        self.mask_mode = mask_mode

    def __len__(self):
        return len(self.dataset)

    def mask_type(self):
        if self.input_mask is None and self.mask_mode:
            mask = (torch.rand(self.im_size) > self.mask_sparsity).int()
        elif self.input_mask is not None and self.mask_mode:
            mask = self.input_mask.int()

        return mask

    def __getitem__(self, idx):
        im, labels = self.dataset[idx]
        im = self.transforms(im)

        if self.mask_mode:
            mask = self.mask_type()
            # im = im * mask
            return im, labels, mask

        return im, labels
