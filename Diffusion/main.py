from diffusers import DDPMPipeline
import torch

# Load a pre-trained model
diffusion = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to("cuda")

# Generate an image
image = diffusion().images[0]
image.show()
