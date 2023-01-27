## Pythonic imports
import sys

sys.path.append("../")
import os
import random
import tqdm
import matplotlib.pyplot as plt
import wandb
from matplotlib.lines import Line2D
import numpy as np

## Torch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam

## Torchvision imports
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torchvision

## Necessary File imports
from networks.classificationNetworks import CNNClassifierNetwork
from networks.samplerNetworks import CNNSamplerNetwork
from data_related_content.data_stuff import MaskedDataset

# wandb.init(project="Restrictive_Sampling_Rewrite")


class TrainingHarness:
    def __init__(self, training_params):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = training_params["classifier"].to(self.device)
        self.sampler = training_params["sampler"].to(self.device)
        self.loops = training_params["loops"]

        self.classifier_optimizer = Adam(self.classifier.backbone.parameters(), lr=0.05)
        self.sampler_optimizer = Adam(self.sampler.parameters(), lr=0.05)
        self.train_dl = training_params["train_dl"]
        self.test_dl = training_params["test_dl"]
        self.criterion = nn.CrossEntropyLoss()

        self.classifier_scheduler = lr_scheduler.OneCycleLR(
            self.classifier_optimizer,
            0.1,
            epochs=training_params["epochs"],
            steps_per_epoch=len(self.train_dl),
            three_phase=True,
        )

        self.sampler_scheduler = lr_scheduler.OneCycleLR(
            self.sampler_optimizer,
            0.1,
            epochs=training_params["epochs"],
            steps_per_epoch=len(self.train_dl) * self.loops,
            three_phase=True,
        )

    def plot_grad_flow(self, named_parameters):
        """Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""

        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend(
            [
                Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4),
            ],
            ["max-gradient", "mean-gradient", "zero-gradient"],
        )

    def save_mask_and_image(self, im_tensor, mask_tensor, loop_no, show=False):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        denorm = torchvision.transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1.0 / s for s in std],
        )
        if mask_tensor is None:
            mask_tensor = torch.rand(im_tensor.shape)
        im_tensor, mask_tensor = (
            im_tensor.detach().cpu(),
            mask_tensor.detach().cpu(),
        )
        im_tensor = denorm(im_tensor)
        im_tensor = torch.clip(im_tensor, 0, 1)
        mask_tensor = torch.clip(mask_tensor, 0, 1)
        im_and_mask = torch.cat((im_tensor, mask_tensor), dim=3)
        grid_image = torchvision.utils.make_grid(im_and_mask, nrow=8)
        grid_image = grid_image.numpy().transpose((1, 2, 0))

        if show:
            plt.imshow(grid_image)
            plt.show()
        plt.title("Original images and masks")
        plt.legend(["Original Image", "Masked Image"])
        plt.imsave(f"{loop_no}_ims.png", grid_image)

    def forward_only_classifier(self, x):
        x = self.classifier(x)
        return x

    def forward_only_sampler(self, x):
        sampler_mask = self.sampler(x)
        return sampler_mask

    def forward_with_sampler(self, x, labels, og_mask):
        sampler_mask = og_mask.unsqueeze(1)
        total_loss = 0.0
        for loop in range(self.loops):
            input_x = x * sampler_mask

            sampler_mask = self.forward_only_sampler(input_x)
            sampler_pred = sampler_mask * x

            out = self.forward_only_classifier(sampler_pred)
            loss = self.criterion(out, labels)
            total_loss += loss

        return total_loss / self.loops, sampler_mask, out

        # return sampler_mask

    def train_only_classifier(self):
        self.classifier.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        for data in tqdm.tqdm(self.train_dl):
            ims, labels, masks = data
            ims, labels, masks = (
                ims.to(self.device),
                labels.to(self.device),
                masks.to(self.device),
            )
            self.classifier_optimizer.zero_grad()
            classifier_out = self.forward_only_classifier(ims * masks.unsqueeze(1))
            loss = self.criterion(classifier_out, labels)

            loss.backward()
            self.plot_grad_flow(self.classifier.named_parameters())

            self.classifier_optimizer.step()
            self.classifier_scheduler.step()
            train_loss += loss.item()
            _, pred = torch.max(classifier_out, 1)
            train_correct += pred.eq(labels).sum().item()
            total += labels.size(0)
        plt.show()
        train_loss /= len(self.train_dl)
        train_acc = (train_correct / total) * 100

        return train_loss, train_acc

    def train_only_sampler(self):
        self.classifier.eval()
        train_loss = 0.0
        train_correct = 0
        total = 0
        for data in tqdm.tqdm(self.train_dl):
            ims, labels, masks = data
            ims, labels, masks = (
                ims.to(self.device),
                labels.to(self.device),
                masks.to(self.device),
            )

            self.sampler_optimizer.zero_grad()
            sampler_loss, sampler_mask, classifier_out = self.forward_with_sampler(
                ims, labels, masks
            )
            sampler_loss.backward()
            self.plot_grad_flow(self.sampler.named_parameters())

            self.sampler_optimizer.step()
            self.sampler_scheduler.step()
            train_loss += sampler_loss.item()
            _, pred = torch.max(classifier_out, 1)
            train_correct += pred.eq(labels).sum().item()
            total += labels.size(0)
        plt.show()

        train_loss /= len(self.train_dl)
        train_acc = (train_correct / total) * 100

        return train_loss, train_acc

    def test_model(self, with_sampler=False):
        if with_sampler:
            self.sampler.eval()
        self.classifier.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data in self.test_dl:
                ims, labels, masks = data
                ims, labels, masks = (
                    ims.to(self.device),
                    labels.to(self.device),
                    masks.to(self.device),
                )
                if with_sampler:
                    (
                        loss,
                        sampler_mask,
                        classifier_out,
                    ) = self.forward_with_sampler(ims, labels, masks)
                else:
                    classifier_out = self.forward_with_classifier(
                        ims * masks.unsqueeze(1)
                    )

                loss = self.criterion(classifier_out, labels)
                _, pred = torch.max(classifier_out, 1)

                test_loss += loss.item()
                test_correct += pred.eq(labels).sum().item()
                test_total += labels.size(0)

        test_loss /= len(self.test_dl)
        test_accuracy = (test_correct / test_total) * 100
        return test_loss, test_accuracy


train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


def train_experiment(sparsity_mask):

    train_params = {}
    train_params["epochs"] = 2
    train_params["batch_size"] = 256
    train_params["is_deterministic"] = False
    train_params["classifier"] = CNNClassifierNetwork()
    train_params["sampler"] = CNNSamplerNetwork()
    train_params["classifier_lr"] = 0.05
    train_params["sampler_lr"] = 0.05
    train_params["mask_sparsity"] = sparsity_mask
    train_params["loops"] = 4

    # set_seed(experiment_seed, is_deterministic=train_params["is_deterministic"])

    if not os.path.exists(f"./{sparsity_mask}_checkpoints"):
        os.makedirs(f"./{sparsity_mask}_checkpoints/")
    if not os.path.exists(f"./{sparsity_mask}_checkpoints/"):
        os.makedirs(f"./{sparsity_mask}_checkpoints/")

    ckpt_path = f"{sparsity_mask}_checkpoints/"

    train_ds = CIFAR10(root="./data", train=True, download=True)
    test_ds = CIFAR10(root="./data", train=False, download=True)

    training_dataset = MaskedDataset(
        train_ds,
        train_params["mask_sparsity"],
        transforms=train_transform,
        im_size=(32, 32),
    )
    test_dataset = MaskedDataset(
        test_ds,
        train_params["mask_sparsity"],
        transforms=test_transform,
        im_size=(32, 32),
    )

    train_dl = DataLoader(
        training_dataset,
        batch_size=train_params["batch_size"],
        shuffle=True,
        num_workers=2,
    )

    test_dl = DataLoader(
        test_dataset,
        batch_size=train_params["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    train_params["train_dl"] = train_dl
    train_params["test_dl"] = test_dl

    training_run = TrainingHarness(training_params=train_params)

    for epoch in range(train_params["epochs"]):
        print(f"Epoch: {epoch}")
        torch.save(
            training_run.classifier.state_dict(),
            f"./{ckpt_path}/classifiermodel_{epoch}.pt",
        )
        torch.save(
            training_run.sampler.state_dict(),
            f"./{ckpt_path}/samplermodel_{epoch}.pt",
        )
        if epoch > 10:
            train_loss, train_acc = training_run.train_only_classifier()
            test_loss, test_acc = training_run.test_model(with_sampler=False)
        else:
            print("Now training both Sampler and Classifier!")
            for param in training_run.classifier.parameters():
                param.requires_grad = False
            classifier_params = sum(
                p.numel()
                for p in training_run.classifier.parameters()
                if p.requires_grad
            )
            sampler_params = sum(
                p.numel() for p in training_run.sampler.parameters() if p.requires_grad
            )
            print(
                f"Trainable Classifier Params: {classifier_params}, Trainable Sampler Params: {sampler_params}"
            )
            train_loss, train_acc = training_run.train_only_sampler()
            test_loss, test_acc = training_run.test_model(with_sampler=True)
        print(
            f"Train Loss: {train_loss}, Train Acc: {train_acc}, Test Loss: {test_loss}, Test Acc: {test_acc}"
        )
        wandb.log(
            {
                "training_loss": train_loss,
                "training_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                # "lr": training_run.scheduler.get_last_lr()[0],
                "epoch": epoch,
            }
        )


if __name__ == "__main__":
    train_experiment(0.5)
