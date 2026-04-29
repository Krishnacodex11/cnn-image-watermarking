import torch
from PIL import Image
from torchvision import transforms
from models import Encoder
from config import *
from utils import denormalize
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ─────────────────────────
encoder = Encoder().to(device)

checkpoint = torch.load("final_model.pth", map_location=device)
encoder.load_state_dict(checkpoint["encoder"])
encoder.eval()

# ── Transform ──────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ── USER INPUT ─────────────────────────
image_path = input("Enter image path (e.g. test.png): ")

wm_input = input(f"Enter {WATERMARK_LENGTH}-bit watermark (e.g. 101010...): ")

# Validate watermark length
if len(wm_input) != WATERMARK_LENGTH:
    raise ValueError(f"Watermark must be {WATERMARK_LENGTH} bits!")

watermark = torch.tensor([[int(i) for i in wm_input]], dtype=torch.float32).to(device)

# ── Load image ─────────────────────────
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# ── Encode ─────────────────────────────
with torch.no_grad():
    wm_img = encoder(img_tensor, watermark)

# ── Save output ────────────────────────
output_path = "watermarked_output.png"

wm_save = denormalize(wm_img[0]).permute(1,2,0).cpu().numpy()

# Convert to PIL-safe format
wm_save = (wm_save * 255).clip(0,255).astype("uint8")
Image.fromarray(wm_save).save(output_path)

# ── Display (for demo) ─────────────────
orig = denormalize(img_tensor[0]).permute(1,2,0).cpu()
wm = denormalize(wm_img[0]).permute(1,2,0).cpu()

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(orig)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Watermarked")
plt.imshow(wm)
plt.axis("off")

plt.show()

print("\n✅ Watermark embedded successfully!")
print(f"Saved as: {output_path}")