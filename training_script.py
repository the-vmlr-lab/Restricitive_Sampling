
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
from models.Sampler_LSTM import SamplerNetwork
from models.Sampler_LSTM import ClassifierNetwork
from TrainingHarnessLstm import Sampler_Classifer_Network_LSTM
from TrainingHarnessVanillaPretrain import Sampler_Classifer_Network_Pretrain


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
    args = parser.parse_args()
    save_path = os.path.join(args.save_path, args.model_name)
    print("hyperparams init")
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print("hyperparams init")
    batch_size       = 64
    epochs           = args.epochs
    mask_per         = args.mask_ratio
    loop_parameter   = args.loop_param
    classifier_start = 1
    print("hyperparams init")
    CIFAR10_dm = CustomCIFAR10DataModule(mask_per * 1024)
    CIFAR10_dm.prepare_data()
    CIFAR10_dm.setup()
    print("setup done")
    sampler_model    = SamplerNetwork(int(mask_per*1024),base_channel_size=32, latent_dim=384)
    classifier_model = ClassifierNetwork()
    main_model       = Sampler_Classifer_Network_Pretrain(sampler_model, classifier_model, loop_parameter, classifier_start, mask_per, save_path)
    trainer          = Trainer(gpus=1, accelerator="gpu", max_epochs=40, profiler='simple')
    trainer.fit(main_model, CIFAR10_dm)
    print("classifier_training_done")
    main_model       = Sampler_Classifer_Network_LSTM(sampler_model, classifier_model, loop_parameter, classifier_start, mask_per, save_path)
    print('model init')
    for name, param in sampler_model.named_parameters():
                print(param.requires_grad)
    trainer          = Trainer(gpus=1, accelerator="gpu",max_epochs=40)
    print("hello")
    trainer.fit(main_model, CIFAR10_dm)

