import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch import nn
from networks import SamplerNetwork,ClassifierNetwork
torch.cuda.is_available()
def train_sampler_loop ( classifier_dataloader,sampler_model,classifier_model,sampler_loss_fn,sampler_optimizer):


  
    next_sampler_data_list = []

    loop_parameter = 1

    for batch, classifier_data in enumerate(classifier_dataloader):
        print(batch)
        classifier_X, classifier_y = classifier_data[0], classifier_data[1]
        sampler_X    = torch.randn(classifier_X.shape).cuda()
        classifier_X = classifier_X.cuda()
        classifier_y = classifier_y.cuda()
        sampler_pred=sampler_X
        loss=0
        for itr in range(0, loop_parameter):


            sampler_pred = sampler_model(sampler_pred)
        # extract pixel_limit values in sample pred or in sample network
        # sampler_pred = torch.einsum('...ij, ...ij -> ...ij'e, sampler_pred, classifier_X)
            filter_out_image=classifier_X*sampler_pred
            classifier_pred = classifier_model(filter_out_image)
            loss += classifier_loss_fn(classifier_pred, classifier_y)

       
        loss.backward()
        sampler_optimizer.step()
          
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor())
    test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor())
    classifier_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
    sampler_model=SamplerNetwork(784)
    classifier_model=ClassifierNetwork()
    sampler_model.cuda()
    classifier_model.cuda()
    learning_rate = 1e-3
    batch_size = 8
    epochs = 2
    sampler_optimizer    = torch.optim.Adam(sampler_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    classifier_optimizer = torch.optim.Adam(classifier_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    sampler_loss_fn = nn.CrossEntropyLoss().cuda() # incorporate number of iterations as a penalty
    classifier_loss_fn = nn.CrossEntropyLoss().cuda()
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_sampler_loop(classifier_dataloader, sampler_model, classifier_model, sampler_loss_fn, sampler_optimizer)
        print("Done!")