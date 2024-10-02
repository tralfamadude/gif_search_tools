import sys
import argparse
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
import time


"""
Image Captioning using BLIP-2.

Model is cached under ~/.cache; to change that, set env var TRANSFORMERS_CACHE.
"""

class BLIP2Wrapper:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        # Define the quantization configuration
        self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    
    def initialize(self, model_name, model_path=None, temperature=1.0, fp16=True, 
                   length_penalty=1.0, search_method="beam", num_beams=5, top_p=0.9):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        start_model_loading = time.time()
        # Load from local path if provided, else from Hugging Face Hub
        self.processor = Blip2Processor.from_pretrained(model_path if model_path else model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_path if model_path else model_name,
            device_map="auto",
            quantization_config=self.quantization_config,
            torch_dtype=torch.float16 if fp16 else torch.float32,
        )

        finish_model_loading = time.time()
        print(f"BLIP2 Model loaded in {finish_model_loading-start_model_loading} sec", file=sys.stderr)

        self.generation_config = {
            "temperature": temperature,
            "length_penalty": length_penalty,
            "num_beams" if search_method == "beam" else "top_p": num_beams if search_method == "beam" else top_p,
        }

        
    def image_to_text(self, images, batch_size=4, max_new_tokens=20):
        results = []
        start_inference = time.time()
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_new_tokens+1,
                early_stopping=True,
                **self.generation_config,
            )
            
            generated_texts = self.processor.batch_decode(generated_ids,
                                                          skip_special_tokens=True,
                                                          clean_up_tokenization_spaces=True)
            results.extend(generated_texts)

        finish_inference = time.time()
        total_time = finish_inference - start_inference
        if len(images) > 0 and total_time > 0.0:
            print(f"BLIP2: Images processed: n={len(images)} total_time={total_time} sec  images_per_sec={len(images)/total_time}", file=sys.stderr)
        return results


    def save_model_to_local(self, model, dest_file_path):
        # Load model and processor from huggingface hub and save to local destination
        print(f"Saving model {model_name} with pretraining {pretraining_name} to {dest_file_path}")
        processor = Blip2Processor.from_pretrained(model)
        model = Blip2ForConditionalGeneration.from_pretrained(model)

        processor.save_pretrained(dest_file_path)
        model.save_pretrained(dest_file_path)
        print(f"Model saved to {dest_file_path}")


    def fetch_images_from_urls(self, image_urls):
        images = []
        for url in image_urls:
            try:
                response = requests.get(url)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error fetching image from {url}: {e}", file=sys.stderr)
        return images

def main():
    parser = argparse.ArgumentParser(
        description="BLIP-2 Image to Text",
        epilog="If both model and model_path are specified, Load model from hugging face, save in local fs"
    )
    parser.add_argument("--model", default="Salesforce/blip2-opt-2.7b", help="BLIP-2 model name")
    parser.add_argument("--model_path", default="", help="directory to local saved BLIP2")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Length penalty")
    parser.add_argument("--search_method", choices=["beam", "nucleus"], default="beam", help="Search method")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p value for nucleus sampling")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate")
    parser.add_argument("images", nargs="+", help="Paths to input images")

    args = parser.parse_args()

    blip2_wrapper = BLIP2Wrapper()

    #
    # special case: if both model and model_path are specified, Load model from hugging face, save in local fs
    #
    if args.model_path != "" and  args.model != "":
        local_path = os.path.expanduser(args.model_path)
        print("Load model from hugging face, save in local fs...")
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        save_model_to_local(args.model, local_path)
        sys.exit(0)
            
    if args.model_path != "":
        print(f"Get BLIP2 from local fs: {args.model_path}")
        blip2_wrapper.initialize(
            model_path=args.model_path,
            temperature=args.temperature,
            fp16=args.fp16,
            length_penalty=args.length_penalty,
            search_method=args.search_method,
            num_beams=args.num_beams,
            top_p=args.top_p
        )
    else:
        print(f"Get BLIP2 from huggingface: {model}")
        blip2_wrapper.initialize(
            model_name=args.model,
            temperature=args.temperature,
            fp16=args.fp16,
            length_penalty=args.length_penalty,
            search_method=args.search_method,
            num_beams=args.num_beams,
            top_p=args.top_p
        )

    if args.images[0].startswith("http"):
        images = fetch_images_from_urls(args.images)
    else:
        images = [Image.open(image_path) for image_path in args.images]
    results = blip2_wrapper.image_to_text(images, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)

    for image_path, result in zip(args.images, results):
        print(f"Image: {image_path}")
        print(f"Description: {result}\n")

if __name__ == "__main__":
    main()
