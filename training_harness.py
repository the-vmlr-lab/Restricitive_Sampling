import torch
from network_definition import VAE
from torchvision import transforms
from torchvision.datasets import CIFAR10
from data_stuff import VAEDataset
from dataset_creation import plot_image_grid
from resnet18_cifar10 import *
import tqdm
import matplotlib.pyplot as plt



training_params = {}
training_params["num_epochs"] = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

ds = CIFAR10(root="./data", train=True)

train_ds = VAEDataset(
    ds,
    "full_train_dataset.parquet",
    transforms=test_transform,
    im_size=(32, 32),
    mask_sparsity=0.5,
)

dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=False)

model = VAE().to(device)
app = resnet18(pretrained=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in tqdm.tqdm(range(400), desc="epochs"):
    for data in tqdm.tqdm(dl, leave=False, desc="batches"):

        random_im, gt_im, labels, matrix_indices = data
        random_im = random_im.to(device).float()
        gt_im = gt_im.to(device).float()
        matrix_indices = matrix_indices.to(device).float()

        optimizer.zero_grad()

        img_rec, mat_rec, z, mu, logvar = model.forward(random_im, matrix_indices)

        im_loss, mat_loss, kld_loss, total_loss = model.vae_loss(recon_image=img_rec,
                          input_image=gt_im,
                          mu=mu,
                          logvar=logvar,
                          mat=mat_rec,
                          input_mat=matrix_indices)
        total_loss.backward()

        optimizer.step()
    if epoch % 10 == 0:
        plot_image_grid(
            [
                random_im[5].permute(1, 2, 0).detach().cpu().numpy(),
                gt_im[5].permute(1, 2, 0).detach().cpu().numpy(),
                img_rec[5].permute(1, 2, 0).detach().cpu().numpy(),
                random_im[10].permute(1, 2, 0).detach().cpu().numpy(),
                gt_im[10].permute(1, 2, 0).detach().cpu().numpy(),
                img_rec[10].permute(1, 2, 0).detach().cpu().numpy(),
                random_im[15].permute(1, 2, 0).detach().cpu().numpy(),
                gt_im[15].permute(1, 2, 0).detach().cpu().numpy(),
                img_rec[15].permute(1, 2, 0).detach().cpu().numpy(),
            ],
            ncols=3,
        )

        out_mask = torch.argmax(app(img_rec), axis=1)
        out_gt = torch.argmax(app(gt_im), axis=1)
        print(f"Ground Truth Label: {labels}")
        print(f"With Predicted Mask Label: {out_mask}")
        print(f"With Ground Truth Mask Label: {out_gt}")
        print(f"acc {(sum(torch.eq(out_mask, labels)) / len(labels)) * 100}")
        print(f"acc {(sum(torch.eq(out_gt, labels)) / len(labels)) * 100}")
        plt.savefig(f"image_file_epoch_{epoch}.png")
        # plt.show()
    # Print the loss every epoch
    print(f"Epoch {epoch+1}/{2}: Loss {loss.item()}")
