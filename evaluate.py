import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from models.unet import UNetFER
from data.dataset import FERDenoiseDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = FERDenoiseDataset("dataset/images", noise_std=0.1)

model = UNetFER().to(device)
model.load_state_dict(torch.load("unet_fer_denoising.pth", map_location=device))
model.eval()

psnr_noisy = []
psnr_denoised = []
ssim_noisy = []
ssim_denoised = []

for noisy, clean in dataset:
    noisy = noisy.unsqueeze(0).to(device)
    clean = clean.unsqueeze(0).to(device)

    with torch.no_grad():
        denoised = model(noisy)

    noisy_np = noisy.cpu().squeeze().numpy()
    denoised_np = denoised.cpu().squeeze().numpy()
    clean_np = clean.cpu().squeeze().numpy()

    psnr_noisy.append(peak_signal_noise_ratio(clean_np, noisy_np))
    psnr_denoised.append(peak_signal_noise_ratio(clean_np, denoised_np))

    ssim_noisy.append(structural_similarity(clean_np, noisy_np, data_range=1.0))
    ssim_denoised.append(structural_similarity(clean_np, denoised_np, data_range=1.0))

print(f"PSNR Noisy     : {np.mean(psnr_noisy):.2f}")
print(f"PSNR Denoised  : {np.mean(psnr_denoised):.2f}")
print(f"SSIM Noisy     : {np.mean(ssim_noisy):.4f}")
print(f"SSIM Denoised  : {np.mean(ssim_denoised):.4f}")
