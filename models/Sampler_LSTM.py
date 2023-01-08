import os, argparse, random
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import lr_scheduler
import torchvision
from torchvision import models
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
import torch.nn.functional as F
from dataloader import CustomCIFAR10DataModule

import wandb
class SamplerNetwork2(nn.Module):
    def __init__(self,k,gpu=True):
        super(SamplerNetwork2, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,

                padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(
              in_channels=64,
              out_channels=32,
              kernel_size=3,
              stride=2,
              padding=1,
              output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size = 3,
                stride=2,
                padding=1, 
                output_padding=1),
          
        )

        self.drop = nn.Dropout2d(0.25)
        print(gpu)
        self.filter=FilterOutMask(k)

    def forward(self, x,y):
        input_shape=x.shape
        out = self.conv_1(x) 
        out = self.conv_2(out)
        out = self.deconv_1(out)
        out = self.deconv_2(out)
        print("here now")
        print(out.shape)
        out=torch.sigmoid(out)
        #out = self.filter(out,y)
        
        return out*y
labels_map = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

class FilterOutMask(nn.Module):
    def __init__(self, sparsity):
        super(FilterOutMask, self).__init__()
        self.sparsity = sparsity

    def forward(self, x,y):
        img_shape = list(x.shape)
        print('Batch Shape', img_shape)
        x_flat = x.view(x.shape[0], -1)
        print('X Flat Shape', x_flat.shape)
        k_idx = int(self.sparsity * x_flat.shape[1])
        print('Kth index', k_idx)
        #threshold = (torch.sort(x_flat, dim=1)[0])[k_idx]
        threshold = torch.sort(x_flat, dim=1)[0][:, k_idx-1]
        print('Threshold', threshold)
        masks = torch.Tensor([])
        masks.requires_grad = True
        for i in range(img_shape[0]):
            mask = (x_flat[i] < threshold[i]).reshape(1, 1, 32, 32)
            masks = torch.cat([masks, mask], 0)
        print(masks.shape)
        masks=masks.view(img_shape)
        print(masks.shape)
        print("here")
        print(y.shape)
        return masks.float()*y

class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*c_hid, latent_dim)
        )

    def forward(self, x):
        return self.net(x)
class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*16*c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, 1, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
             # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x
class SamplerNetwork(nn.Module):
    def __init__(self, k,  base_channel_size: int,
                 latent_dim: int,
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder,
                 num_input_channels: int = 3,
                 width: int = 32,
                 height: int = 32,gpu=True):
        super(SamplerNetwork, self).__init__()
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        self.rnn=nn.LSTMCell(latent_dim,latent_dim).cuda()
        self.latent_dim=latent_dim
        print(gpu)
        print(k)
        self.filter = FilterOutMask(k)

    def forward(self, x,h,c):
        input_shape = x.shape
        z = self.encoder(x)

        
        h_new,c_new=self.rnn(z, (h, c))
        x_hat = self.decoder(h_new)
        out = torch.sigmoid(x_hat)
        
        out = out.view(-1, input_shape[2], input_shape[3])
        return out,h_new,c_new
class ClassifierNetwork(nn.Module):
    def __init__(self):
        super(ClassifierNetwork, self).__init__()

        self.model_final = models.resnet18(pretrained=True)

        self.model_final.conv1   = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_final.maxpool = nn.Identity()
        self.model_final.fc      = nn.Linear(self.model_final.fc.in_features, 10)

    def forward(self, x):
        out = self.model_final(x)
        out = torch.nn.functional.log_softmax(out, dim=1)
        return out