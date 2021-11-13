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
torch.cuda.is_available()
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

class TrainSamplerClassifier(object):
    def __init__(self,classifier_dataset,sampler_model,classifier_model,classifier_loss_fn,sampler_optimizer,test_dataset,classifier_optimizer,loop_parameter,context,save_path):
        self.classifier_dataset=classifier_dataset
        self.sampler_model=sampler_model
        self.classifier_model=classifier_model
        self.classifier_loss_fn=classifier_loss_fn
        self.sampler_optimizer=sampler_optimizer
        self.loop_parameter=loop_parameter
        self.test_dataset=test_dataset
        self.epoch=0
        self.classifier_optimizer=classifier_optimizer
        self.batch_size=64
        self.classifier_start=0.25
        self.context=context
        self.save_path=save_path
        self.test_dataset=test_dataset
    def save_sampler_and_classifier(self):

        torch.save(self.classifier_model.state_dict(), os.path.join(self.save_path,'class.pkl'))
        torch.save(self.sampler_model.state_dict(), os.path.join(self.save_path,'samp.pkl'))

    def visualize_and_save(self,filename,dataset,no_samples=5):

        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        num_batches = len(dataloader)
        sample_no=0
        figure = plt.figure(figsize=(8, 8))
        rows,cols = no_samples, self.loop_parameter+1
        
   
        with torch.no_grad():
            for X, y in dataloader:
                figure.add_subplot(rows,cols, sample_no*cols+1)
                plt.axis("off")
                plt.imshow(X.squeeze(), cmap="gray")
                images, labels = X.to(device), y.to(device)
            
                test = Variable(images.view(-1, 1, 28, 28))
                sampler_X    = Variable(torch.randn(test.size()).to(device))
                sampler_pred=sampler_X
                for itr in range(0, self.loop_parameter):
                    if self.context:
                        sampler_pred=sampler_pred*test
                    sampler_pred = self.sampler_model(sampler_pred)
                    
                    filter_out_image=test*sampler_pred
                    outputs= self.classifier_model(filter_out_image)
                    predictions = torch.max(outputs, 1)[1].to(device)
                    
                    figure.add_subplot(rows,cols, sample_no*cols+itr+2)
                    plt.title(str(labels_map[int(predictions.detach().cpu().numpy().squeeze())]))
                    plt.axis("off")
                    plt.imshow(sampler_pred.detach().cpu().numpy().squeeze())
               
                sample_no+=1
                if sample_no>no_samples-1:
                    break
        plt.savefig(os.path.join(self.save_path,filename)) 



    def test_loop_sampler(self):
        size = len(self.test_dataset)
       
        total, correct = 0, 0
        dataloader = DataLoader(self.test_dataset, batch_size=64)
        num_batches = len(dataloader)
        with torch.no_grad():
            for X, y in dataloader:
                images, labels = X.to(device), y.to(device)
            
                test = Variable(images.view(-1, 1, 28, 28))
                sampler_X    = Variable(torch.randn(test.size()).to(device))
                sampler_pred=sampler_X
                for itr in range(0, self.loop_parameter):
                    if self.context:
                        sampler_pred=sampler_pred*test
                    sampler_pred = self.sampler_model(sampler_pred)
                    filter_out_image=test*sampler_pred
                    outputs= self.classifier_model(filter_out_image)
                   
            
                predictions = torch.max(outputs, 1)[1].to(device)
                correct += (predictions == labels).sum()
                total += len(labels)
            accuracy = correct * 100 / total
        print("Test Accuracy: {}%".format(accuracy))

    def train(self, num_epochs):
        epoch_start_time = time.time()
        print(self.loop_parameter)
        while num_epochs is None or self.epoch < num_epochs:
            self.classifier_model.train()
            self.sampler_model.train()
            self._run_epoch(self.classifier_dataset)
            self.visualize_and_save('train_epoch_'+str(self.epoch)+'.png',self.classifier_dataset)
            self.classifier_model.eval()
            self.sampler_model.eval()
            self.test_loop_sampler()
            self.visualize_and_save('test_epoch_'+str(self.epoch)+'.png',self.test_dataset)
            self.epoch+=1
        self.save_sampler_and_classifier()
        #self.eval_epoch
        #self.evaluate_samplesloop_param
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
                data_X, data_y = classifier_data[0].to(device), classifier_data[1].to(device)
                classifier_X = Variable(data_X.view(-1,1, 28, 28))
                classifier_y= Variable(data_y)
                sampler_X    = Variable(torch.randn(classifier_X.size()).to(device))
                sampler_pred=sampler_X
                loss=0
                for itr in range(0, self.loop_parameter):
                    if self.context:
                        sampler_pred=sampler_pred*classifier_X
                    
                    sampler_pred = self.sampler_model(sampler_pred)
                    filter_out_image=classifier_X*sampler_pred
                    classifier_pred = self.classifier_model(filter_out_image)
                    loss += self.classifier_loss_fn(classifier_pred, classifier_y)

                predictions = torch.max(classifier_pred, 1)[1].to(device)
                correct = (predictions == classifier_y).sum()
                accuracy = correct / classifier_X.size()[0]
                
                if sampler_train:
                    self.sampler_optimizer.zero_grad()
                    loss.backward()
                    self.sampler_optimizer.step()
                elif classifier_train:
                    self.classifier_optimizer.zero_grad()
                    loss.backward()
                    self.classifier_optimizer.step()

                tepoch.set_postfix(loss=loss.data, accuracy=100. * accuracy)
                sleep(0.1)
                self.iter_in_epoch += 1
            

          
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparam initializer')
    parser.add_argument('-m', type=float, dest='mask_ratio', default=1.0,
                        help='mask ratio')
    parser.add_argument('-lp', dest='loop_param', type=int, default=5,
                        help='loop parameter')
    parser.add_argument('-context', dest='context', type=int, default=0,
                        help='context')
    parser.add_argument('-save_folder', dest='save_path', type=str, default='',
                        help='path to folder where to save objects')
    parser.add_argument('-model_name', dest='model_name', type=str, default='baseline',
                        help='name of model')
    args = parser.parse_args()
    save_path=os.path.join(args.save_path,args.model_name)
    if not os.path.exists(save_path):

        os.mkdir(save_path)
    context=True if args.context==1 else False
    classifier_data = datasets.FashionMNIST(root="data",train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))
    test_data = datasets.FashionMNIST(root="data", train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))
    #classifier_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    #eval_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
   
    learning_rate = 1e-3
    batch_size = 64
    epochs = 10
    mask_per = args.mask_ratio
    loop_parameter = args.loop_param
    sampler_model=SamplerNetwork(int(mask_per*784))
    print(mask_per)
    print(context)
    classifier_model=ClassifierNetwork()
    sampler_model.to(device)
    classifier_model.to(device)
    sampler_optimizer    = torch.optim.Adam(sampler_model.parameters(), lr=learning_rate)
    classifier_optimizer = torch.optim.Adam(classifier_model.parameters(), lr=learning_rate)
    classifier_loss_fn = nn.CrossEntropyLoss()
    
    trainer=TrainSamplerClassifier(classifier_dataset=classifier_data,sampler_model=sampler_model,classifier_model=classifier_model,classifier_loss_fn=classifier_loss_fn,sampler_optimizer=sampler_optimizer,test_dataset=test_data,classifier_optimizer=classifier_optimizer,loop_parameter=loop_parameter,context=context,save_path=save_path)
    
    trainer.visualize_and_save('before_training'+'.png',trainer.test_dataset)
    trainer.train(epochs)
 
