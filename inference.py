import torch
import cv2
import numpy as np
from models.unet import UNetFER

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNetFER().to(device)
model.load_state_dict(torch.load("unet_fer_denoising.pth", map_location=device))
model.eval()

img = cv2.imread("dataset/images/img_0.png", cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32) / 255.0
img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    denoised = model(img)

output = denoised.squeeze().cpu().numpy() * 255
cv2.imwrite("denoised.png", output)

print("Denoised image saved.")
