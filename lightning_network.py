import os, argparse, random
import torch
import matplotlib.pyplot as plt
from torch import nn
import torchvision
from torchvision import models
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
import torch.nn.functional as F
from dataloader import CustomCIFAR10DataModule
import wandb

class FilterOutMask(nn.Module):
    """Implemeted using conv layers"""
    def __init__(self, k):

        super(FilterOutMask, self).__init__()
        self.k   = k

    def forward(self, output_a):
        output_flat = torch.flatten(output_a, start_dim=1)
        top_k, top_k_indices = torch.topk(output_flat, self.k, 1)
        mask = torch.zeros(output_flat.shape)
        mask = mask.type_as(output_a)
        src  = torch.ones(top_k_indices.shape)
        src  = src.type_as(output_a)
        mask = mask.scatter(1, top_k_indices, src)
        return mask

class SamplerNetwork(nn.Module):
    def __init__(self, k):
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
                out_channels=1,
                kernel_size = 3,
                stride=2,
                padding=1, 
                output_padding=1),
            nn.ReLU()
        )

        self.drop   = nn.Dropout2d(0.25)
        self.filter = FilterOutMask(k)

    def forward(self, x):
        input_shape = x.shape
        out = self.conv_1(x) 
        out = self.conv_2(out)
        out = self.deconv_1(out)
        out = self.deconv_2(out)
        out = self.filter(out)
        out = out.view(-1, input_shape[2], input_shape[3])
        return out

class ClassifierNetwork(nn.Module):
    def __init__(self):
        super(ClassifierNetwork, self).__init__()
        
        self.model_final = models.resnet18(pretrained=True)
        num_input_fc = self.model_final.fc.in_features
        self.model_final.fc = nn.Linear(num_input_fc, 10)
        '''
        self.model_final = torchvision.models.resnet18(pretrained=True, num_classes=10)
        self.model_final.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_final.maxpool = nn.Identity()
        '''
    def forward(self, x):
        out = self.model_final(x)
        return out

class Sampler_Classifer_Network(LightningModule):
    def __init__(self, sampler, classifier, loop_parameter, classifier_start, mask_per):
        super().__init__()
        self.sampler          = sampler
        self.classifier       = classifier
        self.loop_parameter   = loop_parameter
        self.classifier_start = classifier_start
        self.mask_per         = mask_per
        self.automatic_optimization = False
        self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        data_X, data_y = batch[0], batch[1]
        classifier_X   = data_X.view(-1,3, 32, 32)
        classifier_y   = data_y
        loss           = 0
        classifier_train = False
        sampler_train = False

        if batch_idx >= int(self.classifier_start * self.trainer.num_training_batches):
            classifier_train = True
        else:
            sampler_train    = True

        sampler_optimizer, classifier_optimizer = self.optimizers()

        sampler_pred = batch[2]
        for itr in range(0, self.loop_parameter):
            sampler_pred     = torch.unsqueeze(sampler_pred, 1) * classifier_X
            sampler_pred     = self.sampler(sampler_pred)
            filter_out_image = torch.unsqueeze(sampler_pred, 1) * classifier_X
            classifier_pred  = self.classifier(filter_out_image)
            loss            += F.cross_entropy(classifier_pred, classifier_y)

        predictions = torch.max(classifier_pred, 1)[1]
        correct     = (predictions == classifier_y).sum()
        accuracy    = correct / classifier_X.size()[0]

        self.log("Training accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)


        if sampler_train:
            sampler_optimizer.zero_grad()
            self.manual_backward(loss)
            sampler_optimizer.step()
        elif classifier_train:
            classifier_optimizer.zero_grad()
            self.manual_backward(loss)
            classifier_optimizer.step()
        
        print(loss)

        return loss

    def training_epoch_end(self, training_step_outputs):
        print(training_step_outputs)
    def configure_optimizers(self):
        sampler_optimizer    = torch.optim.Adam(self.sampler.parameters(), lr=1e-3)
        classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-5)
        return sampler_optimizer, classifier_optimizer

    def validation_step(self, batch, batch_idx):
        X, y = batch[0], batch[1]
        test = X.view(-1, 3, 32, 32)
        sampler_pred = batch[2]

        for itr in range(0, self.loop_parameter):
            sampler_pred     = torch.unsqueeze(sampler_pred, 1)  * test
            sampler_pred     = self.sampler(sampler_pred)
            filter_out_image = torch.unsqueeze(sampler_pred, 1)  * test
            outputs          = self.classifier(filter_out_image)

        predictions = torch.max(outputs, 1)[1]
        correct     = (predictions == y).sum()
        validation_accuracy    = correct / X.size()[0]

        self.log("Validation accuracy", validation_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def validation_epoch_end(self, validation_step_outputs):
        return None

    def test_step(self, batch, batch_idx):
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparam initializer')
    parser.add_argument('-m', type=float, dest='mask_ratio', default=1.0,
                        help = 'mask ratio')
    parser.add_argument('-lp', dest='loop_param', type=int, default=5,
                        help = 'loop parameter')
    parser.add_argument('-context', dest='context', type=int, default=0,
                        help = 'context')
    parser.add_argument('-save_folder', dest='save_path', type=str, default='',
                        help = 'path to folder where to save objects')
    parser.add_argument('-model_name', dest='model_name', type=str, default='baseline',
                        help = 'name of model')
    parser.add_argument('-pretrained_classifier', dest='pre_clr', type=int, default=0,
                        help = 'pretrained_classifier')
    parser.add_argument('-epochs', dest='epochs', type=int, default=10,
                    help = 'epochs')
    # args = parser.parse_args()
    # save_path = os.path.join(args.save_path,args.model_name)

    batch_size       = 64
    epochs           = 10  # args.epochs
    mask_per         = 0.5 # args.mask_ratio
    loop_parameter   = 5   #args.loop_param
    classifier_start = 0.25

    CIFAR10_dm = CustomCIFAR10DataModule()
    CIFAR10_dm.prepare_data()
    CIFAR10_dm.setup()

    sampler_model    = SamplerNetwork(int(0.5*1024))
    classifier_model = ClassifierNetwork()
    main_model       = Sampler_Classifer_Network(sampler_model, classifier_model, loop_parameter, classifier_start, mask_per)
    trainer          = Trainer(gpus=1, accelerator="gpu", max_epochs=epochs)
    trainer.fit(main_model, CIFAR10_dm)