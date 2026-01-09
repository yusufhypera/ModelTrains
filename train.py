import torch
from torch.utils.data import DataLoader
from models.unet import UNetFER
from data.dataset import FERDenoiseDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = FERDenoiseDataset("dataset/images", noise_std=0.1)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = UNetFER().to(device)

criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 30

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        output = model(noisy)
        loss = criterion(output, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "unet_fer_denoising.pth")
print("Model saved.")
