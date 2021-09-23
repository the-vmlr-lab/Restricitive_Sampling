import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch import nn
import torch.utils.data as td
from time import sleep
import time
from tqdm import tqdm
from networks import SamplerNetwork,ClassifierNetwork
torch.cuda.is_available()
class TrainSamplerClassifier(object):
    def __init__(self,classifier_dataset,sampler_model,classifier_model,sampler_loss_fn,sampler_optimizer,test_dataset,classifier_optimizer):
        self.classifier_dataset=classifier_dataset
        self.sampler_model=sampler_model
        self.classifier_model=classifier_model
        self.sampler_loss_fn=sampler_loss_fn
        self.sampler_optimizer=sampler_optimizer
        self.loop_parameter=1
        self.test_dataset=test_dataset
        self.epoch=0
        self.classifier_optimizer=classifier_optimizer
        self.batch_size=64
        self.classifier_start=0

    def train(self, num_epochs):
        epoch_start_time = time.time()
        while num_epochs is None or self.epoch < num_epochs:
            self.classifier_model.train()
            self.sampler_model.train()
            self._run_epoch(self.classifier_dataset)
            self.epoch+=1
        #self.eval_epoch
        #self.evaluate_samples
    def _run_epoch(self,dataset,eval=False):
        self.iters_per_epoch = int(len(dataset)/self.batch_size)
        self.iter_starttime = time.time()
        self.iter_in_epoch = 0
        if eval:
            dataloader = td.DataLoader(dataset, batch_size=self.batch_size)
        else:

            dataloader = td.DataLoader(dataset, batch_size=self.batch_size,shuffle=True)
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Sampler Epoch {self.epoch}")
            for classifier_data in tepoch:
                classifier_train=False
                sampler_train=False
                if self.iter_in_epoch==int(self.classifier_start*self.iters_per_epoch):
                    tepoch.set_description(f"Classifier Epoch {self.epoch}")
                    classifier_train=True
                elif self.iter_in_epoch>int(self.classifier_start*self.iters_per_epoch):
                    classifier_train=True

                else:
                    sampler_train=True
                classifier_X, classifier_y = classifier_data[0], classifier_data[1]
                sampler_X    = torch.randn(classifier_X.shape).cuda()
                classifier_X = classifier_X.cuda()
                classifier_y = classifier_y.cuda()
                sampler_pred=sampler_X
                loss=0
                for itr in range(0, self.loop_parameter):

                    sampler_pred = self.sampler_model(sampler_pred)
        # extract pixel_limit values in sample pred or in sample network
        # sampler_pred = torch.einsum('...ij, ...ij -> ...ij'e, sampler_pred, classifier_X)
                    filter_out_image=classifier_X*sampler_pred
                    classifier_pred = self.classifier_model(filter_out_image)
                    loss += classifier_loss_fn(classifier_pred, classifier_y)

                predictions = classifier_pred.argmax(dim=1, keepdim=True).squeeze()
                correct = (predictions == classifier_y).sum().item()
                accuracy = correct / self.batch_size
                loss.backward()
                if sampler_train:
                    self.sampler_optimizer.zero_grad()
                    self.sampler_optimizer.step()
                elif classifier_train:
                    self.classifier_optimizer.zero_grad()
                    self.classifier_optimizer.step()

                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
                sleep(0.1)
                self.iter_in_epoch += 1
            




# def train_sampler_loop ( classifier_dataloader,sampler_model,classifier_model,sampler_loss_fn,sampler_optimizer):


  
#     next_sampler_data_list = []

#     loop_parameter = 1

#     for batch, classifier_data in enumerate(classifier_dataloader):
#         classifier_X, classifier_y = classifier_data[0], classifier_data[1]
#         sampler_X    = torch.randn(classifier_X.shape).cuda()
#         classifier_X = classifier_X.cuda()
#         classifier_y = classifier_y.cuda()
#         sampler_pred=sampler_X
#         loss=0
#         for itr in range(0, loop_parameter):


#             sampler_pred = sampler_model(sampler_pred)
#         # extract pixel_limit values in sample pred or in sample network
#         # sampler_pred = torch.einsum('...ij, ...ij -> ...ij'e, sampler_pred, classifier_X)
#             filter_out_image=classifier_X*sampler_pred
#             classifier_pred = classifier_model(filter_out_image)
#             loss += classifier_loss_fn(classifier_pred, classifier_y)

       
#         loss.backward()
#         if sel
#         sampler_optimizer.step()
          
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    classifier_data = datasets.FashionMNIST(root="data",train=True,download=True,transform=ToTensor())
    test_data = datasets.FashionMNIST(root="data", train=False,download=True,transform=ToTensor())
    #classifier_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    #eval_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    plt.imshow(test_data[0][0].detach().numpy().squeeze(),cmap='gray')
    plt.show()

    sampler_model=SamplerNetwork(784)
    classifier_model=ClassifierNetwork()
    sampler_model.cuda()
    classifier_model.cuda()
    learning_rate = 1e-3
    batch_size = 8
    epochs = 2
    sampler_optimizer    = torch.optim.Adam(sampler_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    classifier_optimizer = torch.optim.Adam(classifier_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    classifier_loss_fn = nn.CrossEntropyLoss().cuda()
    trainer=TrainSamplerClassifier(classifier_data,sampler_model,classifier_model,classifier_loss_fn,sampler_optimizer,test_data,classifier_optimizer)
    trainer.train(2)
  
