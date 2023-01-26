import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np


class CNNHarness(nn.Module):
    def __init__(self, classifier, sampler, loops):
        super(CNNHarness, self).__init__()
        self.classifier = classifier
        self.sampler = sampler
        self.loops = loops

    def save_mask_and_image(self, im_tensor, mask_tensor, loop_no, show=False):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        denorm = torchvision.transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1.0 / s for s in std],
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

    def loopdeedoodle(self, x):
        sampler_mask = self.sampler(x)
        return sampler_mask

    def forward_with_sampler(self, x, og_mask=None):
        im_before_loop = x.clone()
        mask_before_loop = og_mask
        self.save_mask_and_image(
            im_tensor=im_before_loop, mask_tensor=mask_before_loop, loop_no="init"
        )
        # mask is what ya boi got from the dl (x, og_mask)
        for loopy_no in range(self.loops):
            prev_mask = torch.zeros(8, 3, 32, 32)
            mask = self.loopdeedoodle(x)
            x = torch.mul(x, mask)

            diff = prev_mask - mask
            print(torch.norm(diff, p=1))
            prev_mask = mask.clone()
            self.save_mask_and_image(im_tensor=x, mask_tensor=mask, loop_no=loopy_no)
        return x

    def forward_with_classifier(self, x):
        x = self.forward_with_sampler(x)
        x = self.classifier(x)
        return x
