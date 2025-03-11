import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import DiffusionPipeline, DDPMScheduler
from src.functions.dataset_utils import PairedImageDataset, make_dynamic_rs_transform


# Replace these paths with your actual directories
blurry_dir = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/20241107_ds/recon_images/"
sharp_dir  = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/20241107_ds/images/"

dataset = PairedImageDataset(blurry_dir, sharp_dir, transform=make_dynamic_rs_transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# --- Load pre-trained SR3 model components ---
# We load the SR3 pipeline from Hugging Face Diffusers. 
# The SR3 model consists of a U-Net and a noise scheduler.
pipe = SR3Pipeline.from_pretrained("CompVis/sr3", revision="fp16", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Extract the U-Net (denoising model) and the scheduler.
model = pipe.unet
scheduler = pipe.scheduler  # DDPMScheduler (or similar)

# Ensure scheduler is in the proper format.
scheduler.set_format("pt")

# --- Optimizer and Loss ---
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# --- Training Loop ---
num_epochs = 5  # Increase as needed

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        # Get paired images: condition (blurry) and target (sharp)
        blur = batch["blur"].to(device)   # Conditioning image
        sharp = batch["sharp"].to(device)  # Ground truth sharp image

        batch_size = sharp.size(0)
        # Sample a random timestep for each sample (0 to scheduler.num_train_timesteps-1)
        t = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()

        # Add noise to the sharp image according to the diffusion forward process.
        noise = torch.randn_like(sharp)
        noisy_sharp = scheduler.add_noise(sharp, noise, t)

        # The model takes the noisy sharp image, the timestep, and the blurry image as condition.
        # It predicts the noise residual.
        noise_pred = model(noisy_sharp, t, encoder_hidden_states=blur).sample

        loss = criterion(noise_pred, noise)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

# --- Inference ---
# To generate a sharp image from a blurry input, you would run the reverse diffusion process.
# Here’s a simplified one-step example for demonstration:
model.eval()
with torch.no_grad():
    # Load a blurry image from your dataset
    sample = next(iter(dataloader))
    blur = sample["blur"].to(device)
    # Initialize a random tensor as the starting point (or use your VAE output if desired)
    init = torch.randn_like(blur).to(device)
    # Choose a fixed timestep (e.g., mid-noise level)
    fixed_t = torch.tensor([scheduler.num_train_timesteps // 2] * blur.size(0), device=device).long()
    # Use the model to predict noise for this timestep
    noise_pred = model(init, fixed_t, encoder_hidden_states=blur).sample
    # Compute the denoised latent (simplified one-step update)
    # Note: a full reverse diffusion process involves iteratively applying the denoising step.
    alpha_bar = scheduler.alphas_cumprod[fixed_t].view(blur.size(0), 1, 1, 1).to(device)
    refined = (init - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
    