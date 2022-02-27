## LR Finder -- Done, Changing the Optimizer -- In progress, DeepSpeed 
## Integrate augmentations before input to sampler
## Save hyperparameters
## Profiler -- Done
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
        out = self.drop(out)
        out = self.filter(out)
        out = out.view(-1, input_shape[2], input_shape[3])
        return out

class ClassifierNetwork(nn.Module):
    def __init__(self):
        super(ClassifierNetwork, self).__init__()
        
        self.model_final = models.resnet18(pretrained=True)
        
        self.model_final.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model_final.maxpool = nn.Identity()
        self.model_final.fc = nn.Linear(self.model_final.fc.in_features, 10)

    def forward(self, x):
        out = self.model_final(x)
        out = torch.nn.functional.log_softmax(out, dim=1)
        return out

class Sampler_Classifer_Network(LightningModule):
    def __init__(self, sampler, classifier, loop_parameter, classifier_start, mask_per, save_path):
        super().__init__()
        self.sampler          = sampler
        self.classifier       = classifier
        self.loop_parameter   = loop_parameter
        self.classifier_start = classifier_start
        self.mask_per         = mask_per
        self.save_path        = save_path
        self.automatic_optimization = False
        self.learning_rate = 1e-3

        wandb.init("Sampler_classifier")
        wandb.watch(self.classifier, log = "all")
        wandb.watch(self.sampler, log = "all")
        self.save_hyperparameters()

    

    def training_step(self, batch, batch_idx):
        data_X, data_y   = batch[0], batch[1]
        sampler_pred     = batch[2]
        classifier_X     = data_X.view(-1,3, 32, 32)
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
        #classifier_optimizer = self.optimizers()
        #classifier_scheduler = self.lr_schedulers()
        for _ in range(0, self.loop_parameter):
            sampler_pred     = torch.unsqueeze(sampler_pred, 1) * classifier_X
            sampler_pred     = self.sampler(sampler_pred)
            filter_out_image = torch.unsqueeze(sampler_pred, 1) * classifier_X
            classifier_pred  = self.classifier(filter_out_image)
            loss            += F.cross_entropy(classifier_pred, classifier_y)
        

        classifier_pred = self.classifier(classifier_X)
        predictions = torch.max(classifier_pred, 1)[1]
        loss            += F.cross_entropy(classifier_pred, classifier_y)
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
        #classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
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
        sampler_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                sampler_optimizer,
                                0.1,
                                epochs=self.trainer.max_epochs,
                                steps_per_epoch=782)
        
        #sampler_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(sampler_optimizer, self.trainer.max_epochs, 0)
        #classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(classifier_optimizer, self.trainer.max_epochs)
                        
        return [sampler_optimizer, classifier_optimizer], [sampler_scheduler, classifier_scheduler]
        #return [classifier_optimizer], [classifier_scheduler]

    def validation_step(self, batch, batch_idx):
        X, y = batch[0], batch[1]
        test = X.view(-1, 3, 32, 32)
        sampler_pred = batch[2]

        for itr in range(0, self.loop_parameter):
            sampler_pred     = torch.unsqueeze(sampler_pred, 1)  * test
            sampler_pred     = self.sampler(sampler_pred)
            filter_out_image = torch.unsqueeze(sampler_pred, 1)  * test
            outputs          = self.classifier(filter_out_image)
        
        outputs = self.classifier(X)
        predictions = torch.max(outputs, 1)[1]
        correct     = (predictions == y).sum()
        validation_accuracy    = correct / X.size()[0]

        self.log("Validation accuracy", validation_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return validation_accuracy

    def validation_epoch_end(self, validation_step_outputs):
        avg_validation_acc = torch.stack([x for x in validation_step_outputs]).mean()
        self.log("Validation_accuracy", avg_validation_acc, on_epoch=True, logger=True)
        wandb.log({"Validation epoch end accuracy": avg_validation_acc})
        #self.visualize_and_save('train_epoch_'+str(self.trainer.current_epoch)+'.png')

    def test_step(self, batch, batch_idx):
        return None

