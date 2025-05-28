import torch
import torch.nn as nn
import torch.nn.functional as F

"""
U‑Net‑style Variational Auto‑Encoder для лог‑Mel‑спектрограмм 1×128×128.
Основные изменения:
• Скип‑соединения (encoder → decoder) сохраняют детали высоких частот.
• InstanceNorm вместо BatchNorm — устойчивее к размеру батча.
• Latent‑code 128‑D; weight_init — Xavier‑uniform.
"""

class ConvBlock(nn.Module):
    """Conv2d → InstanceNorm → LeakyReLU"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UpBlock(nn.Module):
    """ConvTranspose2d upsample + ConvBlock для refine"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.refine = ConvBlock(out_ch * 2, out_ch)  # после concat

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.refine(x)

class UNetVAE(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # -------- Encoder --------
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 64×64
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = ConvBlock(32, 64)  # 64×64 → conv no stride
        self.down2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)  # 32×32

        self.enc3 = ConvBlock(64, 128)
        self.down3 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)  # 16×16

        self.enc4 = ConvBlock(128, 256)
        self.down4 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)  # 8×8

        # flatten
        self.flatten_dim = 256 * 8 * 8
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_z = nn.Linear(latent_dim, self.flatten_dim)

        # -------- Decoder --------
        self.up1 = UpBlock(256, 128)   # 8→16
        self.up2 = UpBlock(128, 64)    # 16→32
        self.up3 = UpBlock(64, 32)     # 32→64
        self.up4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)  # 64→128

        self._init_weights()

    # ---------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        s1 = self.enc1(x)              # 32×64×64
        s2 = F.leaky_relu(self.down2(self.enc2(s1)), 0.2)  # 64×32×32
        s3 = F.leaky_relu(self.down3(self.enc3(s2)), 0.2)  # 128×16×16
        s4 = F.leaky_relu(self.down4(self.enc4(s3)), 0.2)  # 256×8×8

        h = s4.flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
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
        out = torch.sigmoid(self.up4(d3))
        return out

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, skips)
        return recon, mu, logvar