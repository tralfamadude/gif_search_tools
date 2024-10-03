import open_clip
import torch
import os
import argparse

def main(model_name, pretrained, save_dir):
    # Load the model from Hugging Face Hub
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained)

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_save_path = os.path.join(save_dir, f"openclip_{model_name.lower().replace('-', '_')}_{pretrained}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved locally at {model_save_path}")

    pp_path = os.path.join(save_dir, "preprocess.pth")
    torch.save(preprocess, pp_path)
    print(f"Preprocessing saved locally under {pp_path}.")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save OpenCLIP model locally.\nThe destination directory will be created if it doesn't exist.")
    parser.add_argument("--model", default="ViT-L-14", help="Model name (default: ViT-L-14)")
    parser.add_argument("--pretrained", default="laion2b_s32b_b82k", help="Pretrained spec (default: laion2b_s32b_b82k)")
    parser.add_argument("--save_dir", default="saved_openclip_model", help="Destination directory for saving the model (default: ~/saved_openclip_model)")

    args = parser.parse_args()

    main(args.model, args.pretrained, os.path.expanduser(args.save_dir))

    
