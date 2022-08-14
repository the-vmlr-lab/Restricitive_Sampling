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
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
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
        out = self.filter(x_hat)
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
class Sampler_Classifer_Network_Pretrain(LightningModule):
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
        rows, cols       = len(visualizer_dataloader), self.loop_parameter+1
        device = torch.device('cuda:0')
        with torch.no_grad():
          for X, y, mask in visualizer_dataloader:
            X = X.to(device)
            y = y.to(device)
            mask = mask.to(device)
            figure.add_subplot(rows,cols, sample_no * cols + 1)
            plt.axis("off")
            plt.imshow(X.cpu().squeeze().permute(1, 2, 0), cmap="BrBG")

            test         = X.view(-1, 3, 32, 32)
            test         = test
            sampler_pred = mask

            for itr in range(0, self.loop_parameter):
              sampler_pred     = torch.unsqueeze(sampler_pred, 1)  * test
              sampler_pred     = self.sampler(sampler_pred)
              filter_out_image = torch.unsqueeze(sampler_pred, 1)  * test
              outputs          = self.classifier(filter_out_image)
              predictions      = torch.max(outputs, 1)[1]

              figure.add_subplot(rows,cols, sample_no * cols + itr + 2)
              plt.title(str(labels_map[int(predictions.detach().cpu().numpy().squeeze())]))
              plt.axis("off")
              plt.imshow(sampler_pred.detach().cpu().squeeze())

            sample_no += 1
        plt.savefig(os.path.join(self.save_path, filename))

    def mask_union(self, pred_list):
        final_pred = pred_list[0]
        for i in range(1, len(pred_list)):
            final_pred = final_pred.logical_or(pred_list[0])

        return final_pred

    def training_step(self, batch, batch_idx):
        data_X, data_y   = batch[0], batch[1]
        sampler_pred     = batch[2]
        sampler_prd_list = []
        classifier_X     = data_X.view(-1, 3, 32, 32)
        classifier_y     = data_y
        loss             = 0
        classifier_train = True
        sampler_train    = False

        #if batch_idx >= int(self.classifier_start * self.trainer.num_training_batches):
            #classifier_train = True
        #else:
            #sampler_train    = True

        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()
        sampler_optimizer, classifier_optimizer = optimizers[0], optimizers[1]
        sampler_scheduler, classifier_scheduler = schedulers[0], schedulers[1]
        #hx = torch.randn(classifier_X.shape[0], self.sampler.latent_dim).cuda() # (batch, hidden_size)
        #cx = torch.randn(classifier_X.shape[0], self.sampler.latent_dim).cuda()
        #for _ in range(0, self.loop_parameter):
            #sampler_pred     = torch.unsqueeze(sampler_pred, 1) * classifier_X
            #sampler_pred,hx,cx     = self.sampler(sampler_pred,hx,cx)
            #filter_out_image = torch.unsqueeze(sampler_pred, 1) * classifier_X
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
            sampler_scheduler.step()
        elif classifier_train:
            classifier_optimizer.zero_grad()
            self.manual_backward(loss)
            classifier_optimizer.step()
            classifier_scheduler.step()

        return {"Accuracy":accuracy, "Loss":loss}

    def training_epoch_end(self, training_step_outputs):
        training_loss = torch.stack([x["Loss"] for x in training_step_outputs]).mean()
        training_accuracy = torch.stack([x["Accuracy"] for x in training_step_outputs]).mean()

        wandb.log({"Training Loss": training_loss, "Epoch": self.trainer.current_epoch})
        wandb.log({"Accuracy": training_accuracy, "Epoch": self.trainer.current_epoch})

    def configure_optimizers(self):
        sampler_optimizer    = torch.optim.SGD(self.sampler.parameters(), lr=1e-3, momentum=0.9)
        classifier_optimizer = torch.optim.SGD(
            self.classifier.parameters(),
            lr=0.05,
            momentum=0.9,
            weight_decay=5e-4,
        )

        classifier_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                classifier_optimizer,
                                0.1,
                                epochs=self.trainer.max_epochs,
                                steps_per_epoch=782)
        sampler_scheduler    = torch.optim.lr_scheduler.OneCycleLR(
                                sampler_optimizer,
                                0.1,
                                epochs=self.trainer.max_epochs,
                                steps_per_epoch=782)

        return [sampler_optimizer, classifier_optimizer], [sampler_scheduler, classifier_scheduler]

    def validation_step(self, batch, batch_idx):
        X, y = batch[0], batch[1]
        test = X.view(-1, 3, 32, 32)
        sampler_pred = batch[2]
        #hx = torch.randn(test.shape[0], self.sampler.latent_dim).cuda() # (batch, hidden_size)
        #cx = torch.randn(test.shape[0], self.sampler.latent_dim).cuda()
        #for itr in range(0, self.loop_parameter):
    
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
        rows, cols       = len(visualizer_dataloader), self.loop_parameter+1
        device = torch.device('cuda:0')
        with torch.no_grad():
          for X, y, mask in visualizer_dataloader:
            X = X.to(device)
            y = y.to(device)
            mask = mask.to(device)
            figure.add_subplot(rows,cols, sample_no * cols + 1)
            plt.axis("off")
            plt.imshow(X.cpu().squeeze().permute(1, 2, 0), cmap="BrBG")

            test         = X.view(-1, 3, 32, 32)
            test         = test
            sampler_pred = mask

            for itr in range(0, self.loop_parameter):
              sampler_pred     = torch.unsqueeze(sampler_pred, 1)  * test
              sampler_pred     = self.sampler(sampler_pred)
              filter_out_image = torch.unsqueeze(sampler_pred, 1)  * test
              outputs          = self.classifier(filter_out_image)
              predictions      = torch.max(outputs, 1)[1]

              figure.add_subplot(rows,cols, sample_no * cols + itr + 2)
              plt.title(str(labels_map[int(predictions.detach().cpu().numpy().squeeze())]))
              plt.axis("off")
              plt.imshow(sampler_pred.detach().cpu().squeeze())

            sample_no += 1
        plt.savefig(os.path.join(self.save_path, filename))

    def mask_union(self, pred_list):
        final_pred = pred_list[0]
        for i in range(1, len(pred_list)):
            final_pred = final_pred.logical_or(pred_list[0])

        return final_pred

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
            sampler_scheduler.step()
        elif classifier_train:
            classifier_optimizer.zero_grad()
            self.manual_backward(loss)
            classifier_optimizer.step()
            classifier_scheduler.step()

        return {"Accuracy":accuracy, "Loss":loss}

    def training_epoch_end(self, training_step_outputs):
        training_loss = torch.stack([x["Loss"] for x in training_step_outputs]).mean()
        training_accuracy = torch.stack([x["Accuracy"] for x in training_step_outputs]).mean()

        wandb.log({"Training Loss": training_loss, "Epoch": self.trainer.current_epoch})
        wandb.log({"Accuracy": training_accuracy, "Epoch": self.trainer.current_epoch})

    def configure_optimizers(self):
        sampler_optimizer    = torch.optim.Adam(self.sampler.parameters(), lr=1e-3)
        classifier_optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=0.05,
            weight_decay=5e-4,
        )

        classifier_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                classifier_optimizer,
                                0.1,
                                epochs=self.trainer.max_epochs,
                                steps_per_epoch=782)
        sampler_scheduler    = torch.optim.lr_scheduler.OneCycleLR(
                                sampler_optimizer,
                                0.1,
                                epochs=self.trainer.max_epochs,
                                steps_per_epoch=782)

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

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    batch_size       = 64
    epochs           = args.epochs
    mask_per         = args.mask_ratio
    loop_parameter   = args.loop_param
    classifier_start = 0.25

    CIFAR10_dm = CustomCIFAR10DataModule(mask_per * 1024)
    CIFAR10_dm.prepare_data()
    CIFAR10_dm.setup()

    sampler_model    = SamplerNetwork(int(mask_per*1024),base_channel_size=32, latent_dim=384)
    classifier_model = ClassifierNetwork()
    main_model       = Sampler_Classifer_Network_Pretrain(sampler_model, classifier_model, loop_parameter, classifier_start, mask_per, save_path)
    trainer          = Trainer(gpus=1, accelerator="gpu", max_epochs=40, profiler='simple')
    trainer.fit(main_model, CIFAR10_dm)
    main_model       = Sampler_Classifer_Network_LSTM(sampler_model, classifier_model, loop_parameter, classifier_start, mask_per, save_path)
    trainer          = Trainer(gpus=1, accelerator="gpu", max_epochs=40, profiler='simple')
    trainer.fit(main_model, CIFAR10_dm)

