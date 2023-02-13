import torch
import numpy as np
import matplotlib.pyplot as plt

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


def generate_mask(im_size, mask_sparsity):
    gt_mask = torch.ones((im_size[0], im_size[1]), dtype=torch.int)
    fixed_length = int(im_size[0] * im_size[1] * mask_sparsity)
    indices = torch.randperm(32 * 32, dtype=torch.long)[:fixed_length]

    gt_mask.view(-1)[indices] = 0
    mask_indices = torch.nonzero(gt_mask == 0)
    # print("gt_mask shape", gt_mask.shape)
    # print("mask_indices shape", mask_indices.shape)
    # new_mask = torch.ones(32, 32)
    # new_indices = (inter_mask_indices * 32).to(torch.int64)
    # new_mask[new_indices[:, 0], new_indices[:, 1]] = 0
    # plot_image_grid([gt_mask.cpu().numpy(), new_mask.cpu().numpy()])
    # masks.append(gt_mask)
    # masks_indices.append(mask_indices)
    # plt.show()
    return gt_mask.cpu().numpy(), mask_indices.cpu().numpy()


train_df = pl.DataFrame(
    columns=[
        ("image_no", pl.Int64),
        ("gt_mask"),
        ("masked_indices"),
    ]
)

train_df = train_df.with_column(
    pl.col("masked_indices").cast(pl.List(pl.List(pl.Int64)), strict=False)
)
train_df = train_df.with_column(
    pl.col("gt_mask").cast(pl.List(pl.List(pl.Int32)), strict=False)
)


for i in tqdm.tqdm(range(1000000)):
    gt_mask, masked_indices = generate_mask((32, 32), 0.5)
    image_dict = {
        "image_no": i,
        "gt_mask": pl.Series(gt_mask).arr,
        "masked_indices": pl.Series(masked_indices).arr,
    }

    img_df = pl.DataFrame(
        image_dict,
        orient="row",
    )

    train_df.extend(img_df)
print(train_df)
train_df.write_parquet("1000000_mask_train_dataset.parquet")
"""
if __name__ == "__main__":
    

    
    
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
    Mask Test
    a = np.random.randint(2, 100, size=(5, 2))
    mask = a.flatten() >= np.sort(a.flatten())[::-1][int(0.5 * len(a.flatten()) - 1)]
    print(a.flatten())
    print(mask.astype(np.uint8))
    print(np.sort(a.flatten())[::-1])"""

# viz = show_cam_on_image(np.array(img) / 255.0, grayscale, use_rgb=False)
# plot_image_grid([img, viz, mask, masked_im], 4)
# plt.show()

# print(train_df.width())"""
