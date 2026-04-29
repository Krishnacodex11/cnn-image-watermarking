
import torch
import torch.nn as nn
from config import *

# ── Encoder ─────────────────────────────
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(WATERMARK_LENGTH, IMAGE_SIZE * IMAGE_SIZE)

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, image, watermark):
        B = image.size(0)

        wm = self.fc(watermark).view(B, 1, IMAGE_SIZE, IMAGE_SIZE)
        x = torch.cat([image, wm], dim=1)

        # 🔥 residual embedding (IMPORTANT)
        return image + 0.1 * self.conv(x)


# ── Decoder ─────────────────────────────
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * IMAGE_SIZE * IMAGE_SIZE, WATERMARK_LENGTH),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
