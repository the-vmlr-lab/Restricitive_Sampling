import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, K):
        super(Encoder, self).__init__()
        ## Image -> Encoder
        self.K = K

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

        self.linear1 = nn.Linear(self.K * 2, 512)
        self.linear2 = nn.Linear(512, 256)

        ## mu an var

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

        return z


class Decoder(nn.Module):
    def __init__(self, K):
        super(Decoder, self).__init__()

        self.K = K
        ## decoder image
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
        self.dec_linear1 = nn.Linear(512, 512)
        self.dec_linear2 = nn.Linear(512, self.K * 2)

    def forward(self, z):
        # x = self.in_layer(z)
        x = z.reshape(-1, 512, 1, 1)
        x = self.transpose_block1(x)
        x = self.transpose_block2(x)
        x = self.transpose_block3(x)
        x = self.transpose_block4(x)
        im = self.transpose_block5(x)

        mat = self.dec_linear1(z)
        mat = self.dec_linear2(mat)

        return im, mat


class AE(nn.Module):
    def __init__(self, K):
        super(AE, self).__init__()
        self.K = K
        self.encoder = Encoder(self.K)
        self.decoder = Decoder(self.K)

    def ae_loss(self, recon_image, input_image, mat, input_mat):
        im_recon_loss = F.mse_loss(recon_image, input_image)
        mat_recon_loss = F.mse_loss(mat.view(-1, self.K, 2), input_mat)

        total_loss = im_recon_loss + mat_recon_loss

        return im_recon_loss, mat_recon_loss, total_loss

    def forward(self, input_image, input_matrix):
        z = self.encoder(input_image, input_matrix)
        im, mat = self.decoder(z)

        return im, mat, z