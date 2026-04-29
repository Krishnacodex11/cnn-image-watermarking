
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from config import *

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def generate_watermark(batch_size):
    return torch.randint(0, 2, (batch_size, WATERMARK_LENGTH)).float()

def denormalize(x):
    return x * 0.5 + 0.5

class WatermarkDataset(Dataset):
    def __init__(self, train=True):
        self.data = datasets.CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, _ = self.data[idx]
        watermark = generate_watermark(1).squeeze(0)
        return image, watermark

def get_dataloader(train=True):
    dataset = WatermarkDataset(train=train)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=train)

