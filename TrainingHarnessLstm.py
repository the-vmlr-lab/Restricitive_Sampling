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
from models.Sampler_LSTM import SamplerNetwork
from models.Sampler_LSTM import ClassifierNetwork
from dataloader import CustomCIFAR10DataModule

import wandb
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
class Sampler_Classifer_Network_LSTM(LightningModule):
    def __init__(self, sampler, classifier, loop_parameter, classifier_start, mask_per, save_path):
        super().__init__()
        self.sampler          = sampler
        self.classifier       = classifier
        self.loop_parameter   = loop_parameter
        self.classifier_start = classifier_start
        self.mask_per         = mask_per
        self.save_path        = save_path
        self.learning_rate    = 1e-3
        self.automatic_optimization = False


        wandb.init("Sampler_classifier")
        wandb.watch(self.classifier, log = "all")
        wandb.watch(self.sampler, log = "all")
        self.save_hyperparameters()

    def visualize_and_save(self, filename):
        visualizer_dataloader = self.trainer.datamodule.visualizer_dataloader()
        sample_no        = 0
        figure           = plt.figure(figsize=(8, 8))
        rows, cols       = len(visualizer_dataloader), self.loop_parameter+2
        device = torch.device('cuda:0')
        with torch.no_grad():
          for X, y, mask in visualizer_dataloader:
            X = X.to(device)
            y = y.to(device)
            mask = mask.to(device)
            figure.add_subplot(rows,cols, sample_no * cols + 1)
            plt.axis("off")
            plt.imshow(X.cpu().squeeze().permute(1, 2, 0), cmap="BrBG")
            figure.add_subplot(rows,cols, sample_no * cols + 2)
            plt.axis("off")
            plt.imshow(mask.detach().cpu().squeeze(), cmap="gray")

            test         = X.view(-1, 3, 32, 32)
            test         = test
            sampler_pred = mask
            hx = torch.randn(test.shape[0], self.sampler.latent_dim).cuda() # (batch, hidden_size)
            cx = torch.randn(test.shape[0], self.sampler.latent_dim).cuda()
            for itr in range(0, self.loop_parameter):
              sampler_pred     = torch.unsqueeze(sampler_pred, 1)  * test
              sampler_pred,hx,cx     = self.sampler(sampler_pred,hx,cx)
              filter_out_image = torch.unsqueeze(sampler_pred, 1)  * test
              outputs          = self.classifier(filter_out_image)

              predictions = torch.max(outputs, 1)[1]
              figure.add_subplot(rows,cols, sample_no * cols + itr + 3)
              plt.title(str(labels_map[int(predictions.detach().cpu().numpy().squeeze())]))
              plt.axis("off")
              plt.imshow(sampler_pred.detach().cpu().squeeze())

            sample_no += 1
        plt.savefig(os.path.join(self.save_path, filename))



    def training_step(self, batch, batch_idx):
        data_X, data_y   = batch[0], batch[1]
        sampler_pred     = batch[2]
        sampler_prd_list = []
        classifier_X     = data_X.view(-1, 3, 32, 32)
        classifier_y     = data_y
        loss             = 0
        classifier_train = False
        sampler_train    = False

        if batch_idx >= int(self.classifier_start * self.trainer.num_training_batches):
            classifier_train = True
        else:
            sampler_train    = True

        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()
        sampler_optimizer, classifier_optimizer = optimizers[0], optimizers[1]
        sampler_scheduler, classifier_scheduler = schedulers[0], schedulers[1]
        hx = torch.randn(classifier_X.shape[0], self.sampler.latent_dim).cuda() # (batch, hidden_size)
        cx = torch.randn(classifier_X.shape[0], self.sampler.latent_dim).cuda()
        for _ in range(0, self.loop_parameter):
            sampler_pred     = torch.unsqueeze(sampler_pred, 1) * classifier_X
            sampler_pred,hx,cx     = self.sampler(sampler_pred,hx,cx)
            filter_out_image = torch.unsqueeze(sampler_pred, 1) * classifier_X
            classifier_pred  = self.classifier(filter_out_image)
            loss            += F.cross_entropy(classifier_pred, classifier_y)

        predictions = torch.max(classifier_pred, 1)[1]
        correct     = (predictions == classifier_y).sum()
        accuracy    = correct / classifier_X.size()[0]

        if sampler_train:
            sampler_optimizer.zero_grad()
            self.manual_backward(loss)
            sampler_optimizer.step()
            #sampler_scheduler.step()
        elif classifier_train:
            classifier_optimizer.zero_grad()
            self.manual_backward(loss)
            classifier_optimizer.step()
            #classifier_scheduler.step()
        step=self.trainer.global_step
        if step%100==0:
            print(sampler_train)
            self.visualize_and_save('train_epoch_'+str(step)+'.png')
        return {"Accuracy":accuracy, "Loss":loss, "LR": torch.tensor(classifier_scheduler.get_last_lr()), "LR2": torch.tensor(sampler_scheduler.get_last_lr())}
    
    def training_epoch_end(self, training_step_outputs):
        training_loss = torch.stack([x["Loss"] for x in training_step_outputs]).mean()
        training_accuracy = torch.stack([x["Accuracy"] for x in training_step_outputs]).mean()
        training_lr = torch.stack([x["LR"] for x in training_step_outputs]).mean()
        training_lr2 = torch.stack([x["LR2"] for x in training_step_outputs]).mean()
        wandb.log({"Training Loss": training_loss, "Epoch": self.trainer.current_epoch})
        wandb.log({"Accuracy": training_accuracy, "Epoch": self.trainer.current_epoch})
        wandb.log({"LR": training_lr, "Epoch": self.trainer.current_epoch})
        wandb.log({"LR2": training_lr, "Epoch": self.trainer.current_epoch})

    def configure_optimizers(self):
        sampler_optimizer    = torch.optim.SGD(self.sampler.parameters(), lr=0.01, momentum=0.9)
        classifier_optimizer = torch.optim.SGD(
            self.classifier.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4,
        )
        classifier_scheduler = torch.optim.lr_scheduler.ExponentialLR(classifier_optimizer,1)
        #                        # classifier_optimizer,
        #                         0.1,
        #                         epochs=self.trainer.max_epochs,
        #                         steps_per_epoch=782)
        sampler_scheduler    = torch.optim.lr_scheduler.ExponentialLR(sampler_optimizer,1) #torch.optim.lr_scheduler.OneCycleLR(
        #                         sampler_optimizer,
        #                         0.1,
        #                         epochs=self.trainer.max_epochs,
        #                         steps_per_epoch=782)

        return [sampler_optimizer, classifier_optimizer], [sampler_scheduler, classifier_scheduler]

    def validation_step(self, batch, batch_idx):
        X, y = batch[0], batch[1]
        test = X.view(-1, 3, 32, 32)
        sampler_pred = batch[2]
        hx = torch.randn(test.shape[0], self.sampler.latent_dim).cuda() # (batch, hidden_size)
        cx = torch.randn(test.shape[0], self.sampler.latent_dim).cuda()
        for itr in range(0, self.loop_parameter):
            sampler_pred     = torch.unsqueeze(sampler_pred, 1)  * test
            sampler_pred,hx,cx     = self.sampler(sampler_pred,hx,cx)
            filter_out_image = torch.unsqueeze(sampler_pred, 1)  * test
            outputs          = self.classifier(filter_out_image)

        predictions = torch.max(outputs, 1)[1]
        correct     = (predictions == y).sum()
        validation_accuracy  = correct / X.size()[0]

        self.log("Validation accuracy", validation_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return validation_accuracy

    def validation_epoch_end(self, validation_step_outputs):
        avg_validation_acc = torch.stack([x for x in validation_step_outputs]).mean()
        self.log("Validation_accuracy", avg_validation_acc, on_epoch=True, logger=True)
        wandb.log({"Validation epoch end accuracy": avg_validation_acc})
        #self.visualize_and_save('train_epoch_'+str(self.trainer.current_epoch)+'.png')

    def test_step(self, batch, batch_idx):
        return None