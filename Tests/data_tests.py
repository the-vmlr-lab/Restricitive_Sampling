import sys

sys.path.append("../")

from data_stuff import MaskedDataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import einops


train_ds = CIFAR10("./data", train=True, download=True)

train_tf = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

batch_size = 8
## Test if Masking works properly

"""test_run_1 = MaskedDataset(
    train_ds,
    0.0,
    train_tf,
    im_size=(3, 32, 32),
)

test_run_dl_1 = torch.utils.data.DataLoader(test_run_1, batch_size=8)
for data in test_run_dl_1:
    i, j, k = data

    fig, axes = plt.subplots(2, 8)
    plt.tight_layout()
    for num in range(8):
        sns.heatmap(i[num, 0, :, :], ax=axes[0, num])
        sns.heatmap(i[num, 1, :, :], ax=axes[0, num])
        sns.heatmap(i[num, 2, :, :], ax=axes[0, num])

        sns.heatmap(k[num, 0, :, :], ax=axes[1, num])
        sns.heatmap(k[num, 1, :, :], ax=axes[1, num])
        sns.heatmap(k[num, 2, :, :], ax=axes[1, num])

    plt.show()
    # plt.imshow(
    #    k.permute(2, 1, 0).numpy() * 255,
    # )
    # plt.show()
    # print(len(torch.nonzero(i.flatten())) / len(i.flatten()))
    # print(k.flatten())
    # print(k)
    break"""

## Test if Inference time mask works properly
"""test_run_2 = MaskedDataset(
    train_ds,
    0.0,
    train_tf,
    im_size=(3, 32, 32),
    # input_mask=einops.repeat(
    #    torch.rand(3, 32, 32) > 0.0, "m n o -> k m n o", k=batch_size
    # ),
    input_mask=(torch.rand(3, 32, 32) > 1.0).int(),
)

test_run_dl_2 = torch.utils.data.DataLoader(test_run_2, batch_size=batch_size)


for data in test_run_dl_2:
    i, j, k = data

    fig, axes = plt.subplots(2, 8)
    plt.tight_layout()
    for num in range(1):
        sns.heatmap(i[num, 0, :, :], ax=axes[0, num])
        sns.heatmap(i[num, 1, :, :], ax=axes[0, num])
        sns.heatmap(i[num, 2, :, :], ax=axes[0, num])

        sns.heatmap(k[num, 0, :, :], ax=axes[1, num])
        sns.heatmap(k[num, 1, :, :], ax=axes[1, num])
        sns.heatmap(k[num, 2, :, :], ax=axes[1, num])

    plt.show()
    # plt.imshow(
    #    k.permute(2, 1, 0).numpy() * 255,
    # )
    # plt.show()
    # print(len(torch.nonzero(i.flatten())) / len(i.flatten()))
    # print(k.flatten())
    # print(k)
    break"""

### Test for no mask mode

"""test_run_3 = MaskedDataset(
    train_ds,
    0.0,
    train_tf,
    im_size=(3, 32, 32),
    mask_mode=False,
)

test_run_dl_3 = torch.utils.data.DataLoader(test_run_3, batch_size=batch_size)

for data in test_run_dl_3:
    i, j = data

    fig, axes = plt.subplots(2, 8)
    plt.tight_layout()
    # for num in range(1):
    # sns.heatmap(i[num, 0, :, :], ax=axes[0, num])
    # sns.heatmap(i[num, 1, :, :], ax=axes[0, num])
    # sns.heatmap(i[num, 2, :, :], ax=axes[0, num])

    # sns.heatmap(k[num, 0, :, :], ax=axes[1, num])
    # sns.heatmap(k[num, 1, :, :], ax=axes[1, num])
    # sns.heatmap(k[num, 2, :, :], ax=axes[1, num])

    plt.show()"""


"""
Takeaways:
- Currently a single random mask is applied which is applied to all the images
- The mask is random at all times, do we want to fix with random seed?
"""
