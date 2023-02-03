from network_definition import VAE
from torchsummary import summary


print(summary(VAE(), input_size=([(3, 32, 32), (500, 2)])))
