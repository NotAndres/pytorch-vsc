import torch
from torch import nn


class VSC(nn.Module):
    def __init__(self, latent_dim, c):
        super(VSC, self).__init__()
        self.latent_dim = latent_dim
        self.c = c

        # Initial channels 3 > 128 > 64 > 32
        # Initial filters 3 > 3 > 3

        # First change 3 > 32 > 64 > 128
        # Filters 3 > 3 > 5

        # Second change 3 > 32 > 64 > 128 > 256
        # Filters 3 > 3 > 5 > 5

        # Encoder
        # self.encoder_conv1 = self.getConvolutionLayer(3, 128)
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # self.encoder_conv2 = self.getConvolutionLayer(128, 64)
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # self.encoder_conv3 = self.getConvolutionLayer(64, 32)
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # self.encoder_conv4 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2)
        # )

        self.flatten = nn.Flatten()

        self.encoder_fc1 = nn.Linear(4608, self.latent_dim)
        self.encoder_fc2 = nn.Linear(4608, self.latent_dim)
        self.encoder_fc3 = nn.Linear(4608, self.latent_dim)
        self.encoder_sigmoid = nn.Sigmoid()

        self.reparam_sigmoid = nn.Sigmoid()

        # Decoder
        self.decoder_fc1 = nn.Sequential(
            nn.Linear(self.latent_dim, 4608),
            nn.ReLU()
        )
        # Reshape to 32x12x12
        self.decoder_upsampler1 = nn.Upsample(scale_factor=(2, 2), mode='nearest')

        self.decoder_deconv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2, 2), mode='nearest')
        )
        # 48x48x64
        self.decoder_deconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2, 2), mode='nearest')
        )

        # self.decoder_deconv3 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=(2, 2), mode='nearest')
        # )

        self.decoder_conv1 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1)
        # 96x96x3

    def encode(self, x):
        x = self.encoder_conv1(x)
        x = self.encoder_conv2(x)
        x = self.encoder_conv3(x)

        x = self.flatten(x)
        mu = self.encoder_fc1(x)
        sigma = self.encoder_fc2(x)
        gamma = self.encoder_fc3(x)
        gamma = self.encoder_sigmoid(gamma)

        return mu, sigma, gamma

    def reparameterize(self, mu, logvar, gamma):
        std = torch.exp(0.5 * logvar)
        # Keeps shape, samples from normal dist with mean 0 and variance 1
        eps = torch.randn_like(std)
        # Uniform dist
        eta = torch.rand_like(std)
        slab = self.reparam_sigmoid(self.c * (eta - 1 + gamma))
        return slab * (mu + eps * std)

    def decode(self, z):
        z = self.decoder_fc1(z)
        z = self.decoder_upsampler1(z.view(-1, 32, 12, 12))
        z = self.decoder_deconv1(z)
        z = self.decoder_deconv2(z)
        recon = self.decoder_conv1(z)
        return recon

    def forward(self, x):
        mu, logvar, gamma = self.encode(x)
        z = self.reparameterize(mu, logvar, gamma)
        return self.decode(z), mu, logvar, gamma

    def update_c(self, c):
        self.c = c


# Gamma = Spike
def loss_function(recon_x, x, mu, logvar, gamma, alpha=0.5, beta=1):
    alpha = torch.tensor(alpha)
    gamma = torch.clamp(gamma, 1e-6, 1 - 1e-6)

    mse = torch.mean(torch.sum((x - recon_x).pow(2), dim=(1, 2, 3)))

    slab = torch.sum((0.5 * gamma) * (1 + logvar - mu.pow(2) - logvar.exp()))
    spike_a = (1 - gamma) * (torch.log(1 - alpha) - torch.log(1 - gamma))
    spike_b = gamma * (torch.log(alpha) - torch.log(gamma))

    spike = torch.sum(spike_a + spike_b)
    slab = torch.sum(slab)
    kld = -1 * (spike + slab)
    loss = mse + kld * beta
    return loss, mse, kld, -slab, -spike
