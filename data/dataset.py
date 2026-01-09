import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class FERDenoiseDataset(Dataset):
    def __init__(self, image_dir, noise_std=0.1):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".png")
        ]
        self.noise_std = noise_std

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0

        clean = torch.tensor(img).unsqueeze(0)

        noise = torch.randn_like(clean) * self.noise_std
        noisy = torch.clamp(clean + noise, 0.0, 1.0)

        return noisy, clean
