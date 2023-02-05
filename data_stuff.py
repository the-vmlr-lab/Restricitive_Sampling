import torch
from torch.utils.data import Dataset
import polars as pl
import numpy as np
from torchvision import transforms


class VAEDataset(Dataset):
    def __init__(self, dataset, df_filepath, transforms, im_size, mask_sparsity):
        self.df = pl.read_parquet(df_filepath)
        self.transforms = transforms
        self.dataset = dataset
        self.im_size = im_size
        self.mask_sparsity = mask_sparsity

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        im, labels = self.dataset[idx]
        inp_im = self.transforms(im)
        random_mask = (torch.rand(self.im_size) > self.mask_sparsity).int()
        im = torch.mul(inp_im, random_mask)
        mask_indices = np.array(
            [i.astype(np.uint8) for i in (self.df[idx, 2]).to_numpy()]
        )
        mask_indices = torch.from_numpy(mask_indices)
        gt_mask = np.zeros(self.im_size)
        for i in mask_indices:
            gt_mask[i[0], i[1]] = 1

        return im, inp_im * gt_mask, labels, mask_indices[0:500, :]


class AEDataset(Dataset):
    def __init__(self, dataset, transforms, im_size, mask_sparsity):
        self.transforms = transforms
        self.dataset = dataset
        self.im_size = im_size
        self.mask_sparsity = mask_sparsity

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, labels = self.dataset[idx]
        inp_im = self.transforms(im)
        gt_mask = torch.zeros((32, 32), dtype=torch.int)
        fixed_length = int(32 * 32 * 0.5)
        indices = torch.randperm(32 * 32, dtype=torch.long)[:fixed_length]
        gt_mask.view(-1)[indices] = 1
        mask_indices = torch.argwhere(gt_mask)

        mask_indices = mask_indices.float()
        mask_indices /= 32
        gt_im = torch.mul(inp_im, gt_mask)

        return inp_im, labels, gt_im, mask_indices
