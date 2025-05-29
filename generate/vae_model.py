import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------- Базовый блок -----------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# ----------- Апсемплирующий блок -----------

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.block = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        return self.block(x)

# ----------- VAE -----------

class UNetVAE(nn.Module):
    def __init__(self, latent_dim=128, in_ch=1):
        super().__init__()
        self.latent_dim = latent_dim

        # --- Encoder ---
        self.enc1 = ConvBlock(in_ch, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.down = nn.Conv2d(256, 512, 4, 2, 1)

        self.flatten_dim = 512 * 8 * 8  # для входа 256x256 → 8x8 на bottleneck
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_z = nn.Linear(latent_dim, self.flatten_dim)

        # --- Decoder ---
        self.dec1 = UpBlock(512, 256)
        self.dec2 = UpBlock(256, 128)
        self.dec3 = UpBlock(128, 64)
        self.dec4 = UpBlock(64, 32)
        self.out = nn.ConvTranspose2d(32, in_ch, 4, 2, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        x = F.avg_pool2d(self.enc1(x), 2)  # 256 → 128
        x = F.avg_pool2d(self.enc2(x), 2)  # 128 → 64
        x = F.avg_pool2d(self.enc3(x), 2)  # 64 → 32
        x = F.avg_pool2d(self.enc4(x), 2)  # 32 → 16
        x = self.down(x)                  # 16 → 8
        h = x.flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_z(z).view(-1, 512, 8, 8)
        x = self.dec1(h)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        return torch.sigmoid(self.out(x))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# ----------- Дискриминатор -----------

class Discriminator(nn.Module):
    def __init__(self, in_ch=1):
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
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(x.size(0))
