import torch
from PIL import Image
from torchvision import transforms
from models import Decoder
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ─────────────────────────
decoder = Decoder().to(device)

checkpoint = torch.load("final_model.pth", map_location=device)
decoder.load_state_dict(checkpoint["decoder"])
decoder.eval()

# ── Transform ──────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ── USER INPUT ─────────────────────────
image_path = input("Enter watermarked image path: ")

# ── Load image ─────────────────────────
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

img_tensor = img_tensor + 0.05 * torch.randn_like(img_tensor)
# ── Decode ─────────────────────────────
with torch.no_grad():
    pred_wm = decoder(img_tensor)

# Convert to binary
pred_bits = (pred_wm > 0.5).int().cpu().numpy()

print("\n🔍 Extracted Watermark:")
print(pred_bits)