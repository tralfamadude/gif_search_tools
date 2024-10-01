import open_clip
import torch
import os
from PIL import Image
import numpy as np

# Check if GPU is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define the local path where the model and preprocess are saved
save_dir = "saved_openclip_model"
model_save_path = os.path.join(save_dir, "openclip_vit_l_14_laion2b_s32b_b82k.pth")
preprocess_save_path = os.path.join(save_dir, "preprocess.pth")

# Load the model architecture (empty model) and move it to the appropriate device
model = open_clip.create_model('ViT-L-14', pretrained=False).to(device)

# Load the state dict from the local file
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()  # Set the model to evaluation mode

# Load the preprocessing transforms
preprocess = torch.load(preprocess_save_path)

# Create a random 32x32 RGB image to simulate an input
random_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
image = Image.fromarray(random_image)

# Preprocess the image
image_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

# Generate embeddings for the image
with torch.no_grad():  # No need to calculate gradients for inference
    image_embedding = model.encode_image(image_tensor)

# Print the embedding and verify its size
print(f"Generated image embedding shape: {image_embedding.shape}")
print("Image embedding:", image_embedding)

