import open_clip
import torch
import os

# Define the model name and Hugging Face hub path
model_name = "ViT-L-14"
pretrained = "laion2b_s32b_b82k"
# Define a local path where the model will be saved
save_dir = os.path.expanduser("saved_openclip_model")

# Load the model from Hugging Face Hub
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained)


# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the model state dict locally
model_save_path = os.path.join(save_dir, "openclip_vit_l_14_laion2b_s32b_b82k.pth")
torch.save(model.state_dict(), model_save_path)

print(f"Model saved locally at {model_save_path}")

pp_path = os.path.join(save_dir, "preprocess.pth")
# You can also save the preprocess function's info, if needed
# (For this example, we assume it is being saved for reuse, but adjust as necessary)
torch.save(preprocess, pp_path)

print(f"Preprocessing saved locally under {pp_path}.")

