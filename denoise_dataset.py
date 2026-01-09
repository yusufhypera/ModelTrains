import os
import cv2
import torch
import numpy as np
from models.unet import UNetFER

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNetFER().to(device)
model.load_state_dict(torch.load("unet_fer_denoising.pth", map_location=device))
model.eval()

INPUT_DIR = "dataset/images"
OUTPUT_DIR = "dataset/images_denoised"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".png"):
        continue

    img = cv2.imread(os.path.join(INPUT_DIR, fname), cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        denoised = model(tensor)

    out = denoised.squeeze().cpu().numpy() * 255
    cv2.imwrite(os.path.join(OUTPUT_DIR, fname), out)

print("All images denoised.")
