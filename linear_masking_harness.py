import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from network_definition import LinearMaskingModule, LinearMaskingModuleCNN
from data_stuff import LinearMaskingDataset
from collections import OrderedDict


def plot_image_grid(images, ncols=None, cmap="gray", column_titles=None):
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
            if column_titles:
                for i, title in enumerate(column_titles):
                    axes[i].set_title(title)


def pixel_wise_accuracy(inp_img_tensor, pred_img_tensor):
    inp_img_tensor = (inp_img_tensor < 0.5).float()
    # pred_img_tensor = inp_img_tensor.clone()
    pred_img_tensor = (pred_img_tensor < 0.5).float()
    return (
        sum(torch.eq(inp_img_tensor, pred_img_tensor)) / inp_img_tensor.size(0)
    ).sum() / (1024)


training_params = {}
training_params["num_epochs"] = 1000
training_params["lr"] = 1.0
training_params["save_every"] = 100
training_params["batch_size"] = 256
training_params["save_path"] = "./experiments/cnn_l2_1"
training_params["resume"] = False
training_params["resume_epoch"] = 200
training_params[
    "resume_path"
] = f"{training_params['save_path']}/ckpts/model_{training_params['resume_epoch']}.pt"
device = torch.device("cpu")

train_ds = LinearMaskingDataset("mask_train_dataset.parquet")

dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)

model = LinearMaskingModule(K=512)

save_path = training_params["save_path"]
if not os.path.exists(save_path):
    os.makedirs(save_path + "/" + "imgs")
    os.makedirs(save_path + "/" + "ckpts")

model = LinearMaskingModule(K=512).float()

optimizer = torch.optim.SGD(model.parameters(), lr=training_params["lr"])
"""optimizer = torch.optim.SGD(
    model.parameters(), lr=training_params["lr"], momentum=0.9, nesterov=True
)"""
# optimizer = torch.optim.Adam(model.parameters(), lr=training_params["lr"])

if training_params["resume"]:
    checkpoint = torch.load(training_params["resume_path"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    new_state_dict = OrderedDict()
    for k, v in checkpoint["model_state_dict"].items():
        name = k[10:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)

model = torch.compile(model, mode="reduce-overhead")
model.to(device)


for epoch in tqdm.tqdm(range(training_params["num_epochs"]), desc="epochs"):
    batch_loss = 0.0
    batch_acc = 0.0
    with tqdm.tqdm(dl, unit="batch", leave=False) as tepoch:
        for data in tepoch:

            gt_mask, matrix_indices = data
            gt_mask, matrix_indices = gt_mask.to(device), matrix_indices.to(device)
            optimizer.zero_grad()

            mask_pred = model(matrix_indices)
            loss = F.l1_loss(mask_pred, gt_mask)
            batch_loss += loss.item()
            acc = pixel_wise_accuracy(gt_mask, mask_pred)
            batch_acc += acc.item()
            loss.backward()
            optimizer.step()
        print(
            f"Epoch {epoch}/{training_params['num_epochs']}: | Total Loss: {batch_loss / (10000 / training_params['batch_size']):4f},  \
             | Total Acc: {batch_acc / (10000 / training_params['batch_size']):4f}"
        )
    """grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    print(grads)"""

    # pdb.set_trace()

    if epoch % 100 == 0:
        plot_image_grid(
            [
                gt_mask[0].detach().cpu().numpy(),
                mask_pred[0].detach().cpu().numpy(),
                gt_mask[10].detach().cpu().numpy(),
                mask_pred[10].detach().cpu().numpy(),
                gt_mask[15].detach().cpu().numpy(),
                mask_pred[15].detach().cpu().numpy(),
            ],
            ncols=2,
            column_titles=["Ground Truth Mask", "Network Prediction"],
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            },
            f"{save_path}/ckpts/model_{epoch}.pt",
        )

        plt.savefig(f"{save_path}/imgs/epoch_{epoch}.png")
