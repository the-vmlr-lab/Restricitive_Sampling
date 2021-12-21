
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch import nn

class FilterOutMask(nn.Module):
    """Implemeted using conv layers"""
    def __init__(self, k,gpu):

        super(FilterOutMask, self).__init__()
        self.k=k
        self.gpu=gpu


    def forward(self, output_a):
        output_flat=torch.flatten(output_a,start_dim=1)
        top_k,top_k_indices=torch.topk(output_flat,self.k,1)
        mask=torch.zeros(output_flat.shape)
        src=torch.ones(top_k_indices.shape)
        if self.gpu:
            mask=mask.cuda()
            src=src.cuda()
        mask = mask.scatter(1, top_k_indices, src)
        return mask

class SamplerNetwork(nn.Module):
    def __init__(self,k,gpu=True):
        super(SamplerNetwork, self).__init__()
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
                out_channels=3,
                kernel_size = 3,
                stride=2,
                padding=1, 
                output_padding=1),
            nn.ReLU()
        )

        self.drop = nn.Dropout2d(0.25)
        print(gpu)
        self.filter=FilterOutMask(k,gpu)

    def forward(self, x):
        input_shape=x.shape
        out = self.conv_1(x) 
        out = self.conv_2(out)
        out = self.deconv_1(out)
        out = self.deconv_2(out)
        out = self.filter(out)
        out=out.view(input_shape)
        return out
class ClassifierNetwork(nn.Module):
    def __init__(self):
        super(ClassifierNetwork, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
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
                padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


        self.fc_1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc_2 = nn.Linear(in_features=600, out_features=120)
        self.fc_3 = nn.Linear(in_features=120, out_features=10)
        

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = out.view(out.size(0), -1)
        out = self.fc_1(out)
        out = self.drop(out)
        out = self.fc_2(out)
        out = self.fc_3(out)
        
        return out
if __name__ == '__main__':
    sampler_model = SamplerNetwork(int(0.5*3*1024),gpu=False)
    print(type(sampler_model))
    ip = torch.rand(1, 3, 32, 32, requires_grad=False)

    out = sampler_model(ip)
    plt.imshow(ip.squeeze().permute(1, 2, 0), cmap="BrBG")
    plt.show()
    plt.clf()
    print(out.detach().numpy().squeeze())
    plt.imshow(out.detach().squeeze().permute(1, 2, 0))
    plt.show()

