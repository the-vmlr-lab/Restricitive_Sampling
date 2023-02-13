import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        ## Image -> Encoder

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # stride=1
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # stride=1
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # stride=1
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # stride=2
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # stride=2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )

        self.conv_linear = nn.Linear(512, 256)

        ## Matrix -> Encoder

        self.linear1 = nn.Linear(500 * 2, 512)
        self.linear2 = nn.Linear(512, 256)

        ## mu an var
        self.linear_mu = nn.Linear(512, 128)
        self.linear_logvar = nn.Linear(512, 128)

    def forward(self, x, y):

        # Image forward

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.conv_linear(x)

        # Matrix forward
        y = torch.flatten(y, start_dim=1)
        y = self.linear1(y)
        y = self.linear2(y)

        ## Combine
        z = torch.cat([x, y], axis=1)

        mu = self.linear_mu(z)
        logvar = self.linear_logvar(z)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        ## decoder image
        self.in_layer = nn.Linear(128, 512)
        self.transpose_block1 = nn.Sequential(
            nn.ConvTranspose2d(
                512, 256, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.transpose_block2 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.transpose_block3 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.transpose_block4 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        self.transpose_block5 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
        ## decode matrix
        self.dec_linear1 = nn.Linear(128, 512)
        self.dec_linear2 = nn.Linear(512, 1000)

    def forward(self, z):
        x = self.in_layer(z)
        x = x.reshape(-1, 512, 1, 1)
        x = self.transpose_block1(x)
        x = self.transpose_block2(x)
        x = self.transpose_block3(x)
        x = self.transpose_block4(x)
        im = self.transpose_block5(x)

        mat = self.dec_linear1(z)
        mat = self.dec_linear2(mat)

        return im, mat


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu

    def vae_loss(self, recon_image, input_image, mu, logvar, mat, input_mat):
        im_recon_loss = F.mse_loss(recon_image, input_image)
        mat_recon_loss = F.mse_loss(mat.view(-1, 500, 2), input_mat)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )

        total_loss = 0.998 * im_recon_loss + 0.001 * mat_recon_loss + 0.00025 * kld_loss

        return im_recon_loss, mat_recon_loss, kld_loss, total_loss

    def forward(self, input_image, input_matrix):
        mu, logvar = self.encoder(input_image, input_matrix)
        z = self.reparameterize(mu, logvar)
        im, mat = self.decoder(z)

        return im, mat, z, mu, logvar
