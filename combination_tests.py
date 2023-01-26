from CombinationHarness import CNNHarness

from classificationNetworks import CNNClassifierNetwork
from samplerNetworks import CNNSamplerNetwork

import torch
from torch.utils.data import DataLoader
from data_stuff import MaskedDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

train_ds = CIFAR10("./data", train=True, download=True)

train_tf = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

batch_size = 8

classifier_model = CNNClassifierNetwork()
sampler_model = CNNSamplerNetwork()

model = CNNHarness(classifier=classifier_model, sampler=sampler_model, loops=15)

test_run_1 = MaskedDataset(
    train_ds,
    0.5,
    train_tf,
    im_size=(3, 32, 32),
)

test_run_dl_1 = torch.utils.data.DataLoader(test_run_1, batch_size=batch_size)

for data in test_run_dl_1:
    images, labels, masks = data
    model.forward_with_sampler(images, masks)
    break
    # print(model.forward_with_classifier(images).shape)
