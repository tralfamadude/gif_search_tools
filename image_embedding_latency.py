import time
import argparse
import torch
import open_clip
from PIL import Image, ImageSequence
import numpy as np

def extract_images_from_gif(gif_path):
    gif = Image.open(gif_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    return frames

def measure_latency(model, preprocess, images, warmup_cycles=2):
    latencies = []

    for i, img in enumerate(images):
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        # Warmup cycles (skipped in the stats)
        if i < warmup_cycles:
            _ = model.encode_image(img_tensor)
            continue

        # Measure latency
        start_time = time.time()
        with torch.no_grad():
            _ = model.encode_image(img_tensor)
        end_time = time.time()

        latencies.append(end_time - start_time)

    return latencies

def main():
    parser = argparse.ArgumentParser(description="Measure latency for image embedding using OpenCLIP")
    parser.add_argument('gif_path', type=str, help="Path to the GIF file")
    parser.add_argument('--model', type=str, default="ViT-B-32", help="OpenCLIP model name")
    parser.add_argument('--pretrain', type=str, default="openai", help="Pretrained weights for the model")

    args = parser.parse_args()

    # Load model and preprocessing
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrain)
    model = model.to(device).eval()

    # Extract images from the GIF
    images = extract_images_from_gif(args.gif_path)

    # Measure latency
    latencies = measure_latency(model, preprocess, images)

    # Calculate stats
    latencies_np = np.array(latencies)
    mean_latency = latencies_np.mean()*1000.
    min_latency = latencies_np.min()*1000.
    max_latency = latencies_np.max()*1000.
    std_latency = latencies_np.std()*1000.

    # Print results
    img_width, img_height = images[0].size
    print("model,width,height,area,mean_latency_ms,min_latency_ms,max_latency_ms,sd_ms")
    print(f"{args.model},{img_width},{img_height},{img_width*img_height},{mean_latency:.3f},{min_latency:.3f},{max_latency:.3f},{std_latency:.3f}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main()

