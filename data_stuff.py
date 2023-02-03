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
