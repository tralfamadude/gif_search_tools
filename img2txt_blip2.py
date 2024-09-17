import argparse
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class BLIP2Wrapper:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None

    def initialize(self, model_name="Salesforce/blip2-opt-2.7b", temperature=1.0, fp16=False, 
                   length_penalty=1.0, search_method="beam", num_beams=5, top_p=0.9):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if fp16 else torch.float32,
            device_map="auto"
        )

        self.model.to(self.device)
        
        self.generation_config = {
            "temperature": temperature,
            "length_penalty": length_penalty,
            "num_beams" if search_method == "beam" else "top_p": num_beams if search_method == "beam" else top_p,
        }

    def image_to_text(self, images, batch_size=4, max_new_tokens=50):
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **self.generation_config
            )
            
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            results.extend(generated_texts)
        
        return results

def main():
    parser = argparse.ArgumentParser(description="BLIP-2 Image to Text")
    parser.add_argument("--model", default="Salesforce/blip2-opt-2.7b", help="BLIP-2 model name")
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
    blip2_wrapper.initialize(
        model_name=args.model,
        temperature=args.temperature,
        fp16=args.fp16,
        length_penalty=args.length_penalty,
        search_method=args.search_method,
        num_beams=args.num_beams,
        top_p=args.top_p
    )

    images = [Image.open(image_path) for image_path in args.images]
    results = blip2_wrapper.image_to_text(images, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)

    for image_path, result in zip(args.images, results):
        print(f"Image: {image_path}")
        print(f"Description: {result}\n")

if __name__ == "__main__":
    main()
