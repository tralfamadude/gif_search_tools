import torch
import open_clip
import json
import argparse

class QueryEmbedding:
    def __init__(self, model_name="ViT-L-14", pretrain="laion2b_s32b_b82k"):
        self.model_name = model_name
        self.pretrain = pretrain
        self.model = None
        self.tokenizer = None
        self.device = None

    def select_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def initialize(self):
        self.device = self.select_device()
        self.model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, self.pretrain, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

    def calculate_embedding(self, text):
        return self.calculate(text, self.model, self.tokenizer, self.device)

    def calculate(self, text, model, tokenizer, device):
        with torch.no_grad():
            tokenized_text = tokenizer([text]).to(device)
            text_features = model.encode_text(tokenized_text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return {"query": text, "embedding": text_features.cpu().numpy().tolist()}

    @staticmethod
    def dict_to_json(embedding_dict):
        return json.dumps(embedding_dict)


# Main function to read input file, process each query, and output the result as JSONL
def main():
    parser = argparse.ArgumentParser(description="Compute OpenCLIP text embeddings.")
    parser.add_argument("input_file", type=str, help="File with one text query per line.")
    parser.add_argument("--model", type=str, default="ViT-L-14", help="Model name for OpenCLIP (default: ViT-L-14)")
    parser.add_argument("--pretrain", type=str, default="laion2b_s32b_b82k", help="Pretraining weights (default: laion2b_s32b_b82k)")
    parser.add_argument("--output_file", type=str, default="output.jsonl", help="Output file (default: output.jsonl)")
    args = parser.parse_args()

    # Initialize the QueryEmbedding class
    query_embedder = QueryEmbedding(args.model, args.pretrain)
    query_embedder.initialize()

    # Process each query and write embeddings to the output file in JSONL format
    with open(args.input_file, "r") as infile, open(args.output_file, "w") as outfile:
        for line in infile:
            text = line.strip()
            if not text:
                continue
            embedding_dict = query_embedder.calculate_embedding(text)
            json_str = QueryEmbedding.embedding_to_json(embedding_dict)
            outfile.write(json_str + "\n")

if __name__ == "__main__":
    main()


