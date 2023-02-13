from data_stuff import *
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset_creation import plot_image_grid

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

ds = CIFAR10(root="./data", train=True)

train_ds = VAEDataset(
    ds,
    "train_dataset.parquet",
    transforms=test_transform,
    im_size=(32, 32),
    mask_sparsity=0.5,
)
print("helo")
for data in train_ds:
    print("helo?")
    random_im, gt_im, labels, mask_indices = data
    print(f"Random Image Shape :{random_im.shape}")
    print(f"Ground Truth Image Shape :{gt_im.shape}")
    print(f"Labels Shape: {labels}")
    print(f"Mask Indices: {mask_indices.shape}")
    # plot_image_grid(
    #    [random_im.permute(1, 2, 0).numpy(), gt_im.permute(1, 2, 0).numpy()]
    # )
    plt.show()
