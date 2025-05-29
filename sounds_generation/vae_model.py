import torch
import torch.nn as nn
import torch.nn.functional as F

"""
UNetVAE‑3ch + Discriminator
• Генератор: VAE
• Дискриминатор: CNN для оценки реалистичности спектрограмм
"""

# ----------- VAE часть -----------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.refine = ConvBlock(out_ch * 2, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.refine(x)

class UNetVAE(nn.Module):
    def __init__(self, latent_dim: int = 128, in_ch: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        # ---- Encoder ----
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = ConvBlock(32, 64)
        self.down2 = nn.Conv2d(64, 64, 4, 2, 1)
        self.enc3 = ConvBlock(64, 128)
        self.down3 = nn.Conv2d(128, 128, 4, 2, 1)
        self.enc4 = ConvBlock(128, 256)
        self.down4 = nn.Conv2d(256, 256, 4, 2, 1)

        self.flatten_dim = 256 * 8 * 8
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_z = nn.Linear(latent_dim, self.flatten_dim)

        # ---- Decoder ----
        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up3 = UpBlock(64, 32)
        self.up4 = nn.ConvTranspose2d(32, in_ch, 4, 2, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        s1 = self.enc1(x)
        s2 = F.leaky_relu(self.down2(self.enc2(s1)), 0.2)
        s3 = F.leaky_relu(self.down3(self.enc3(s2)), 0.2)
        s4 = F.leaky_relu(self.down4(self.enc4(s3)), 0.2)
        h = s4.flatten(1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar, (s1, s2, s3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, skips):
        s1, s2, s3 = skips
        h = self.fc_z(z).view(-1, 256, 8, 8)
        d1 = self.up1(h, s3)
        d2 = self.up2(d1, s2)
        d3 = self.up3(d2, s1)
        return torch.sigmoid(self.up4(d3))

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, skips)
        return recon, mu, logvar

# ----------- Дискриминатор -----------

class Discriminator(nn.Module):
    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 0),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(x.size(0))

