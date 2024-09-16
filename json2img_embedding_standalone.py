import sys
import os
import json
import torch
import base64
from PIL import Image
from io import BytesIO
import open_clip
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from scipy.spatial.distance import cosine
from collections import defaultdict
from codetiming import Timer
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
#from line_profiler import LineProfiler # temporary

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # supress warnings if clusters < k for Kmeans



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"------->Device is {device}<------", file=sys.stderr)

def process_jsonl_line(jsonl_string):
    data = json.loads(jsonl_string)
    hash_value = data.get("hash")
    gif_base64 = data.get("gifb64")
    gif_binary = base64.b64decode(gif_base64)
    return hash_value, gif_binary

def extract_images_from_gif(gif_binary, hash=""):
    images = []
    try:
        with Image.open(BytesIO(gif_binary)) as img:
            for frame in range(0, img.n_frames):
                img.seek(frame)
                images.append(img.convert("RGB"))
    except OSError as message:
        print(f"Corrupted gif base64 on hash={hash}", file=sys.stderr)
    except Exception as message: 
        print(f"Corrupted_image conversion on hash={hash}", file=sys.stderr)
    return images


def extract_hashes_from_jsonl(file_path):
    hashes_set = set()  # Set to store unique hashes
    
    # Open the jsonl file and process each line
    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Parse each line as JSON
                data = json.loads(line)
                # Add the hash value to the set if it exists
                if 'hash' in data:
                    hashes_set.add(data['hash'])
            except json.JSONDecodeError:
                # Handle JSON parsing error if any line is not valid JSON
                print(f"Error parsing jsonl line")
    
    # Return the set of unique hashes
    return hashes_set


def most_different_vectors(vector_list, k=3, threshold=0.99):
    """
    Selects up to k diverse vectors using the Maxmin algorithm. 
    Complexity: O(kn)
    
    Args:
    vector_list (list): List of numpy arrays or lists, each representing a vector
    k (int): Maximum number of vectors to select
    threshold (float): Cosine similarity threshold. If all remaining vectors are within
                       this threshold of similarity to selected vectors, stop selecting.
    
    Returns:
    list of numpy arrays: List of up to `k` most different vectors.
    """
    n_vectors = len(vector_list)
    
    # Convert list of vectors to 2D numpy array
    vectors = np.array([np.array(v) for v in vector_list])
    
    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms
    
    # Initialize with a random vector
    selected = [np.random.randint(n_vectors)]
    
    while len(selected) < k:
        # Compute cosine similarities between selected and all vectors
        similarities = np.dot(normalized_vectors, normalized_vectors[selected].T)
        
        # Find the maximum similarity for each vector to any selected vector
        max_similarities = np.max(similarities, axis=1)
        
        # Find the vector with the minimum max similarity (i.e., most diverse)
        candidate_idx = np.argmin(max_similarities)
        
        # Check if the candidate is diverse enough
        if 1 - max_similarities[candidate_idx] <= threshold:
            # If not diverse enough, stop selecting
            break
        
        selected.append(candidate_idx)
    
    return [vector_list[i] for i in selected]


def process_input_files(input_files, model_name, pretrained, output_file, k=3, neighborhood_threshold=0.1):
    # Initialize OpenCLIP model and preprocessing
    start_model_loading = time.time()
    print("Initialize OpenCLIP model with pretraing", file=sys.stderr)
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained)
    model = model.to(device).eval()
    model.half(); print("precision=fp16")  ### fp16 ####
    finish_model_loading = time.time()
    print(f"Model loaded in {finish_model_loading-start_model_loading} sec", file=sys.stderr)
    
    if os.path.exists(output_file):
        previously_processed = extract_hashes_from_jsonl(output_file)
    else:
        previously_processed = set()

    # Prepare for writing output
    output_jsonl = open(output_file, "a") if output_file else sys.stdout
    vector_count_by_hash = defaultdict(int)
    print(f"Input file count={len(input_files)}", file=sys.stderr)
    
    embedding_dimensions = 0
    total_gifs_processed = 0
    total_images_processed = 0
    total_embeddings_saved = 0
    start_processing = time.time()

    # Process each input file
    for input_file in input_files:
        with open(input_file, "r") as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue

                # Process the line to extract hash and GIF binary
                hash_value, gif_binary = process_jsonl_line(line)

                # skip furher processing if it has already been done
                if hash_value in previously_processed:
                    continue


                images = extract_images_from_gif(gif_binary, hash=hash_value)
                if len(images) == 0:
                    continue

                total_gifs_processed += 1
                embeddings = []

                for image in images:
                    total_images_processed += 1
                    image_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension
                    with torch.no_grad():
                        image_embedding = model.encode_image(image_tensor).squeeze(0)  # Remove batch dimension
                        embeddings.append(image_embedding.cpu().numpy())

                # select representative vectors to reduce the number of embeddings 
                selected_embeddings = most_different_vectors(embeddings, top_n=k, threshold=(1.0 - neighborhood_threshold))
                # print(f"{hash_value}: {len(selected_embeddings)}")

                total_embeddings_saved += len(selected_embeddings)
                if embedding_dimensions == 0:
                    embedding_dimensions = len(selected_embeddings[0])
                for embedding in selected_embeddings:
                    output = {
                        "hash": hash_value,
                        "embedding": embedding.tolist(),  # Convert tensor to list for JSON
                        "mspec": f"{model_name}/{pretrained}"
                    }
                    print(json.dumps(output), file=output_jsonl)
                    vector_count_by_hash[hash_value] += 1

    finish_processing = time.time()
    total_time = finish_processing-start_processing
    print(f"Total gifs={total_gifs_processed} images={total_images_processed} embeddings_saved={total_embeddings_saved}")
    print(f"total_time_secs={total_time}")
    if total_gifs_processed > 0:
        print(f"gifs/sec={total_gifs_processed/total_time}")
        print(f"images per gif={total_images_processed/total_gifs_processed}  embeddings per gif={total_embeddings_saved/total_gifs_processed}")

    if output_file:
        output_jsonl.close()

    # Calculate and print histogram
    counts = list(vector_count_by_hash.values())
    max_count = max(counts) if counts else 1
    histogram = [counts.count(i) / len(counts) for i in range(1, max_count + 1)]
    print("Histogram of vector counts by hash (normalized):", histogram)

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process images in GIFs, generate embeddings using OpenCLIP, and output results.")
    parser.add_argument("input_files", nargs="+", help="Input JSONL files containing image data")
    parser.add_argument("--output_file", required=True, help="Destination file for JSONL output")
    parser.add_argument("--model_name", default="ViT-B-32", help="Model name for OpenCLIP (default: ViT-B-32)")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k", help="Pretrained weights for OpenCLIP (default: laion2b_s34b_b79k)")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters/vectors to select (default: 3)")
    parser.add_argument("--neighborhood_threshold", type=float, default=0.1, help="Minimum cosine distance between vectors (default: 0.1)")

    args = parser.parse_args()

    #profiler = LineProfiler()
    #profiler.add_function(process_input_files)
    #profiler.add_function(extract_images_from_gif)
    #profiler.add_function(extract_hashes_from_jsonl)
    #profiler.add_function(most_different_vectors)
    #profile_wrapper = profiler(process_input_files)
    # Process input files and output results
    #profile_wrapper(args.input_files, args.model_name, args.pretrained, args.output_file, args.k, args.neighborhood_threshold)
    process_input_files(args.input_files, args.model_name, args.pretrained, args.output_file, args.k, args.neighborhood_threshold)
    #profiler.print_stats()

