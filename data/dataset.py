import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class FERDenoiseDataset(Dataset):
    
    def __init__(self, image_dir, noise_std=0.1):
        self.image_dir = image_dir
        self.noise_std = noise_std
        self.image_files = [
            f for f in os.listdir(image_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.image_files.sort()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        
        # Load image as grayscale and normalize to [0, 1]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        
        # Add Gaussian noise
        noise = np.random.randn(*img.shape).astype(np.float32) * self.noise_std
        noisy_img = np.clip(img + noise, 0, 1)
        
        # Convert to tensors with channel dimension
        clean = torch.tensor(img).unsqueeze(0)
        noisy = torch.tensor(noisy_img).unsqueeze(0)
        
        return noisy, clean
