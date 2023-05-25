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

class TrainClassifier(object):
    def __init__(self,classifier_dataset,classifier_model,classifier_loss_fn,test_dataset,classifier_optimizer,save_path):
        self.classifier_dataset   = classifier_dataset
        self.classifier_model     = classifier_model
        self.classifier_loss_fn   = classifier_loss_fn
        self.test_dataset         = test_dataset
        self.epoch                = 0
        self.classifier_optimizer = classifier_optimizer
        self.batch_size           = 64
        self.save_path            = save_path

    def train(self, num_epochs):
        while self.epoch < num_epochs:
            self.classifier_model.train()
            self._run_epoch(self.classifier_dataset)
            self.test_classifer()
            self.epoch += 1


    def _run_epoch(self, dataset, eval=False):
        self.iters_per_epoch = int(len(dataset)/self.batch_size)
        self.iter_starttime  = time.time()
        self.iter_in_epoch   = 0
        self.epoch_loss      = 0
        self.accuracy        = 0

        if eval:
            dataloader = td.DataLoader(dataset, batch_size = self.batch_size, shuffle = False)
        else:
            dataloader = td.DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        with tqdm(dataloader, unit='batch') as tepoch:
            tepoch.set_description("Classifier Epoch {}".format(self.epoch))
            for classifier_data in tepoch:
                data_X, data_y = classifier_data[0].to(device), classifier_data[1].to(device)
                classifier_X   = Variable(data_X.view(-1, 1, 28, 28))
                classifier_y   = Variable(data_y)

                classifier_pred = self.classifier_model(classifier_X)
                loss = self.classifier_loss_fn(classifier_pred, classifier_y)

                predictions   = torch.max(classifier_pred, 1)[1].to(device)
                correct       = (predictions == classifier_y).sum()
                self.accuracy = correct / len(predictions) * 100

                self.classifier_optimizer.zero_grad()
                loss.backward()
                self.classifier_optimizer.step()

                self.epoch_loss = loss.data

                tepoch.set_postfix(loss = loss.data, accuracy = self.accuracy)
                self.iter_in_epoch += 1

        writer.add_scalar("Training Loss", self.epoch_loss, self.epoch)
        writer.add_scalar("Accuracy", self.accuracy, self.epoch)

        if self.epoch % 3 == 0:
            SAVE_PATH = self.save_path + '/Epoch_' + str(self.epoch) + '.pth'
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.classifier_model.state_dict(),
                'optimizer_state_dict': self.classifier_optimizer.state_dict(),
                'loss': self.epoch_loss
            }, SAVE_PATH)

    def test_classifer(self):
        dataloader  = DataLoader(self.test_dataset, batch_size=64)
        num_batches = len(dataloader)
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                images, labels = X.to(device), y.to(device)

                test = Variable(images.view(-1, 1, 28, 28))
                outputs = self.classifier_model(images)

                predictions = torch.max(outputs, 1)[1].to(device)
                correct += (predictions == labels).sum()
                total   += len(labels)
        
        accuracy = correct * 100 / total
        writer.add_scalar("Eval mode", accuracy, self.epoch)
        print("Test Accuracy: {}%".format(accuracy))  


writer = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    context = True if args.context==1 else False
    classifier_data = datasets.FashionMNIST(root="data",train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))
    test_data = datasets.FashionMNIST(root="data", train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))
    #classifier_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    #eval_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
   
    learning_rate = 1e-3
    batch_size = 64
    epochs = 10
    mask_per = args.mask_ratio
    loop_parameter = args.loop_param

    classifier_model = ClassifierNetwork()
    classifier_model.to(device)
    classifier_optimizer = torch.optim.Adam(classifier_model.parameters(), lr=learning_rate)
    classifier_loss_fn   = nn.CrossEntropyLoss()
    
    trainer = TrainClassifier(classifier_dataset=classifier_data, classifier_model=classifier_model, classifier_loss_fn=classifier_loss_fn, test_dataset=test_data, classifier_optimizer=classifier_optimizer, save_path=save_path)
    trainer.train(epochs)