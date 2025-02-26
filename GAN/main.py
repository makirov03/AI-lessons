import torch
import torch.nn as nn
import torch.optim as optim
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1, 28, 28)


# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x.view(-1, 784))


# Training Setup
def train_gan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataloader = DataLoader(torchvision.datasets.MNIST(root=".", train=True, transform=transform, download=True),
                            batch_size=64, shuffle=True)

    for epoch in range(10):
        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            optimizer_d.zero_grad()
            outputs = discriminator(real_images)
            loss_real = criterion(outputs, real_labels)

            z = torch.randn(batch_size, 100).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            loss_fake = criterion(outputs, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            outputs = discriminator(fake_images)
            loss_g = criterion(outputs, real_labels)
            loss_g.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch + 1}/10], Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")

    # Save the trained generator
    torch.save(generator.state_dict(), "generator.pth")
    print("Model saved as generator.pth")


def generate_and_show():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    generator.load_state_dict(torch.load("generator.pth", map_location=device))
    generator.eval()

    z = torch.randn(16, 100).to(device)
    fake_images = generator(z).cpu().detach()

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(fake_images[i].squeeze(), cmap="gray")
        ax.axis("off")
    plt.show()


# Comment out training to avoid running it repeatedly
# train_gan()

generate_and_show()
