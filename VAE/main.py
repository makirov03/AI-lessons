import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 20)  # Mean and variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x.view(-1, 784))
        return self.decoder(x[:, :10]).view(-1, 1, 28, 28)


def train_vae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    dataloader = DataLoader(datasets.MNIST("./", transform=transforms.ToTensor(), download=True), batch_size=64,
                            shuffle=True)
    for epoch in range(5):
        for images, _ in dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = vae(images)
            loss = ((images - outputs) ** 2).mean()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/5], Loss: {loss.item()}")


train_vae()
