import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import nn
import torch.utils.data as td
from time import sleep
import time
from tqdm import tqdm
from networks import SamplerNetwork,ClassifierNetwork
from torch.autograd import Variable
import argparse
import os
from torch.utils.tensorboard import SummaryWriter

mask_per=0.05
loop_parameter=5
context=True
model='models/results/best_m0.05_l5_context'
test_dataset = datasets.FashionMNIST(root="data", train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_loop_sampler():
        size = len(test_dataset)
        print(size)
        total, correct = 0, 0
        flipped_correct=0
        dataloader = DataLoader(test_dataset, batch_size=64)
        num_batches = len(dataloader)
        with torch.no_grad():
            for X, y in dataloader:
                images, labels = X.to(device), y.to(device)
            
                test = Variable(images.view(-1, 1, 28, 28))
                sampler_X    = Variable(torch.randn(test.size()).to(device))#pick some random n amounts pixels
                sampler_pred=sampler_X
                predictions_temp=None
                for itr in range(0,loop_parameter):
                    if context:
                        sampler_pred=sampler_pred*test
                    sampler_pred = sampler_model(sampler_pred)
                    filter_out_image=test*sampler_pred
                    outputs= classifier_model(filter_out_image)
                    if itr==0:

                        predictions_temp=torch.max(outputs, 1)[1].to(device)
                predictions = torch.max(outputs, 1)[1].to(device)
                correct += (predictions == labels).sum()
                flipped_correct+=((predictions_temp!=predictions) & (predictions == labels) ).sum()
                total += len(labels)
            accuracy = correct * 100 / total
        print(flipped_correct)
        print("Test Accuracy: {}%".format(accuracy))
sampler_model=SamplerNetwork(int(mask_per*784))
sampler_model.to(device)
sampler_model.load_state_dict(torch.load(os.path.join(model,"samp.pkl")))
classifier_model=ClassifierNetwork()
classifier_model.to(device)
classifier_model.load_state_dict(torch.load(os.path.join(model,"class.pkl")))
sampler_model.eval()
classifier_model.eval()
test_loop_sampler()
