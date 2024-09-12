import torch
import clip
import open_clip
import json
import argparse

# Function to select device (CPU or GPU)
def select_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# Function to load the OpenCLIP model based on the model name and pretraining
def load_model(model_name="ViT-B-32", pretrain="laion400m_e32"):
    device = select_device()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrain, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer, device

# Function to calculate the text embeddings using the OpenCLIP model
def calculate(text, model, tokenizer, device):
    with torch.no_grad():
        tokenized_text = tokenizer([text]).to(device)
        text_features = model.encode_text(tokenized_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return {"query": text, "embedding": text_features.cpu().numpy().tolist()}

# Wrapper function to return the JSON string from the calculate function's output
def embedding_to_json(embedding_dict):
    return json.dumps(embedding_dict)

# Main function to read input file, process each query, and output the result as JSONL
def main():
    parser = argparse.ArgumentParser(description="Compute OpenCLIP text embeddings.")
    parser.add_argument("input_file", type=str, help="File with one text query per line.")
    parser.add_argument("--model", type=str, default="ViT-L-14", help="Model name for OpenCLIP (default: ViT-L-14)")
    parser.add_argument("--pretrain", type=str, default="laion2b_s32b_b82k", help="Pretraining weights (default: laion2b_s32b_b82k)")
    parser.add_argument("--output_file", type=str, default="output.jsonl", help="Output file (default: output.jsonl)")
    args = parser.parse_args()

    # Load the model
    model, tokenizer, device = load_model(args.model, args.pretrain)

    # Process each query and write embeddings to the output file in JSONL format
    with open(args.input_file, "r") as infile, open(args.output_file, "w") as outfile:
        for line in infile:
            text = line.strip()
            if not text:
                continue
            embedding_dict = calculate(text, model, tokenizer, device)
            json_str = embedding_to_json(embedding_dict)
            outfile.write(json_str + "\n")

if __name__ == "__main__":
    main()
