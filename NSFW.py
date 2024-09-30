import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import time
import sys
import argparse
from typing import List
import os
import numpy as np

"""
Class to process images with Falconsai nsfw model to get a score for images [0.0, 1.0].

Model is from huggingface.co.
"""

class NSFW:
    def __init__(self, batch_size: int = 32, use_fp16: bool = False):
        self.model_name = "Falconsai/nsfw_image_detection"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.image_processor = None
        self.model = None

        
    def initialize(self):
        """
        Load model and processor. 
        Returns: None.
        """
        start_time = time.time()
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        # Debugging lines
        print("Image Processor Configuration:", file=sys.stderr)
        print(f"{self.image_processor}", file=sys.stderr)
        print(f"Image Mean: {self.image_processor.image_mean}", file=sys.stderr)
        print(f"Image Std: {self.image_processor.image_std}", file=sys.stderr)

        self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        if self.use_fp16 and self.device.type == "cuda":
            self.model.half()
            print(f"nsfw model: fp16={self.use_fp16}  device={self.device}", file=sys.stderr)
        self.model.eval()
        finish_time = time.time()
        print(f"nsfw model loaded in {finish_time - start_time} seconds", file=sys.stderr)


    def process_images(self, images: List[Image.Image], hash:str) -> List[float]:
        """
        Process given list of images and return a corresponding list of floats. 
        images: list of PIL.Image
        hash: associated file or id used only for error messages. 
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        start_time = time.time()
        # Prepare inputs
        try: 
            inputs = self.image_processor(images, return_tensors="pt")
        except Exception as message:
            print("\n", file=sys.stderr)
            for idx, img in enumerate(images):
                img_np = np.array(img)
                print(f"NSFW.process_images(): image {idx + 1}/{len(images)} | Shape: {img_np.shape}", file=sys.stderr)
            print(f"NSFW.process_images(hash={hash}): {message}", file=sys.stderr)
            print("\n", file=sys.stderr)
            return [0.0]

        if self.use_fp16:
            inputs = {k: v.half() for k, v in inputs.items()}
        inputs = {k: v.to(self.device) for k, v in inputs.items()} # move to gpu if using it

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process outputs
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        nsfw_scores = probabilities[:, 1].tolist()  # Assuming NSFW is the second class

        finish_time = time.time()
        total_time = finish_time - start_time
        avg_time_per_image = total_time / len(images)
        #print(f"Average nsfw processing time per image: {avg_time_per_image:.4f} seconds", file=sys.stderr)
        return nsfw_scores

    
    def process_image_paths(self, image_paths: List[str]) -> List[float]:
        """
        Convenience function to process a list of images in the file system. 
        Useful for testing. 
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        num_images = len(image_paths)

        results = []
        total_time = 0

        # Process images in batches
        for i in range(0, num_images, self.batch_size):
            batch_paths = image_paths[i:i+self.batch_size]
            batch_images = [Image.open(path).convert("RGB") for path in batch_paths]
            
            batch_results = self.process_images(batch_images)
            
            results.extend(batch_results)
            
        return results

    
def main():
    parser = argparse.ArgumentParser(description="NSFW Image Detection")
    parser.add_argument("input_files", nargs="+", help="Input image files")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    args = parser.parse_args()

    nsfw_detector = NSFW(batch_size=args.batch_size, use_fp16=args.fp16)
    nsfw_detector.initialize()

    results = nsfw_detector.process_image_paths(args.input_files)

    with open(args.output, "w") as f:
        for path, score in zip(args.input_files, results):
            f.write(f"{path}: {score:.6f}\n")

if __name__ == "__main__":
    main()
