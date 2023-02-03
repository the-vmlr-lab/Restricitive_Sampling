import torch
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
from resnet18_cifar10 import *
import matplotlib.pyplot as plt

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import polars as pl
import tqdm

device = torch.device("cpu")


def plot_image_grid(images, ncols=None, cmap="gray"):
    """Plot a grid of images"""
    if not ncols:
        factors = [i for i in range(1, len(images) + 1) if len(images) % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1
    nrows = int(len(images) / ncols) + int(len(images) % ncols)
    imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
    f, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows))
    axes = axes.flatten()[: len(imgs)]
    for img, ax in zip(imgs, axes.flatten()):
        if np.any(img):
            # if len(img.shape) > 2 and img.shape[2] == 1:
            #    img = img.squeeze()
            ax.imshow(img, cmap=cmap)


def generate_mask(img, label, model, retention_percent=0.5):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    targets = [ClassifierOutputTarget(label)]

    grayscale_cam = cam(input_tensor=img, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    mask = (
        grayscale_cam.flatten()
        <= np.sort(grayscale_cam.flatten())[::-1][
            int(retention_percent * len(grayscale_cam.flatten()) - 1)
        ]
    ).reshape(32, 32)

    index_matrix = np.argwhere(mask)
    # index_dict = {f"{i}": list(index) for i, index in enumerate(index_matrix[0:5])}
    masked_im = (img[0] * mask).permute(1, 2, 0).numpy()
    return img, mask, index_matrix, grayscale_cam, masked_im


if __name__ == "__main__":
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    ds = CIFAR10(root="./data", train=True, transform=test_transform)
    train_df = pl.DataFrame(
        columns=[
            ("image_no", pl.Int64),
            ("label", pl.Int64),
            ("masked_indices"),
        ]
    )
    train_df = train_df.with_column(
        pl.col("masked_indices").cast(pl.List(pl.List(pl.Int64)), strict=False)
    )

    dl = torch.utils.data.DataLoader(ds, batch_size=1)
    model = resnet18(pretrained=True)

    for index, data in enumerate(tqdm.tqdm(dl)):
        img_in, label = data
        img_out, mask, index_matrix, grayscale, masked_im = generate_mask(
            img_in, label, model
        )
        image_dict = {
            "image_no": index,
            "label": label.numpy(),
            "masked_indices": pl.Series(index_matrix).arr,
        }

        img_df = pl.DataFrame(
            image_dict,
            orient="row",
        )

        train_df.extend(img_df)

    train_df.write_parquet("full_train_dataset.parquet")
    """Mask Test
    a = np.random.randint(2, 100, size=(5, 2))
    mask = a.flatten() >= np.sort(a.flatten())[::-1][int(0.5 * len(a.flatten()) - 1)]
    print(a.flatten())
    print(mask.astype(np.uint8))
    print(np.sort(a.flatten())[::-1])"""

    # viz = show_cam_on_image(np.array(img) / 255.0, grayscale, use_rgb=False)
    # plot_image_grid([img, viz, mask, masked_im], 4)
    # plt.show()

    # print(train_df.width())
