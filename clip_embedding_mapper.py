#!/usr/bin/env python3
import sys
import os
import json
import torch
import torch.cuda
import base64
from PIL import Image
from io import BytesIO
import open_clip
from collections import defaultdict
import time
import numpy as np
import img2txt_blip2
import keyword_extractor
import re
import nltk
from nltk.corpus import stopwords



"""
Process jsonl from stdin with json (hash, gifb64) that is, a hash or other unique identifier and 
a base64 encoded GIF file. 

Output (hash, embedding, keywords)  where keywords is present only if BLIP2 is enabled.
The output will be 1-k embedding lines per input gif. (k default is 3.)

The normalized embeddings are suitable for vector (semantic) search when queries are processed by query_to_vector.py using the same OpenCLIP model.

The jsonl output goes to stdout. 
Info about performance goes to stderr, including messages of hashes that correspond to corrupted GIFs.

Models are from huggingface.co, but we move them to local fs, dir is specified on command line.

"""


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"------->Device is {device}<------", file=sys.stderr)

skip_img2txt = True  # skip BLIP2 if True

"""
The 2.7g BLIP2 (image->text) model is slow. Expect 0.5 seconds per image on GPU. 
"""

class KeywordExtractor:
    def __init__(self):
        self.setup_nltk()
        self.stop_words = set(stopwords.words('english'))

    def setup_nltk(self):
        nltk_data_dir = os.path.expanduser('/tmp/nltk')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        nltk.data.path.append(nltk_data_dir)

        try:
            stopwords.words('english')
        except LookupError:
            print("Downloading NLTK stopwords...", file=sys.stderr)
            nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)

    def extract_keywords(self, text):
        # Convert to lowercase and remove punctuation
        words = re.findall(r'\w+', text.lower())
        
        # Remove stop words and create a set of unique words
        keywords = [word for word in words if word not in self.stop_words]
        
        return list(set(keywords))


def process_jsonl_line(jsonl_string):
    """
    jsonl_string: a json string for a dict, with keys "hash" (could be file name, whatever) and "gifb64" (base64).
    Returns: (hash, gif-binary) 
    """
    data = json.loads(jsonl_string)
    hash_value = data.get("hash")
    gif_base64 = data.get("gifb64")
    gif_binary = base64.b64decode(gif_base64)
    return hash_value, gif_binary


def extract_images_from_gif(gif_binary, hash=""):
    """
    Extract images from a GIF.
    git_binary: the bytes of a GIF file.
    hash: optional associated hash, only for error messages. 
    Returns: list of images (PIL Image) from the given GIF. 
    """
    images = []
    try:
        with Image.open(BytesIO(gif_binary)) as img:
            for frame in range(0, img.n_frames):
                img.seek(frame)
                images.append(img.convert("RGB"))
    except OSError as message:
        print(f"Corrupted: gif base64 on hash: {hash}", file=sys.stderr)
    except Exception as message: 
        print(f"Corrupted: image conversion on hash: {hash}", file=sys.stderr)
    return images

#ToDo: skip extract_hashes_from_jsonl()

def validate_image(image: Image.Image, min_width: int = 6, min_height: int = 6, depth: int = 3) -> (bool, int, int):
    """
    Validates a PIL Image based on minimum width, height, and number of channels. Some images are not
    worth processing because they are too small to represent something. Example: animated separators 
    which are 1 pixel high. 
            
    Parameters:
    - image (PIL.Image.Image): The image to validate.
    - min_width (int): The minimum required width of the image in pixels. Default is 10.
    - min_height (int): The minimum required height of the image in pixels. Default is 10.
    - depth (int): The required number of channels in the image. Default is 3 (e.g., RGB).
    return: bool: True if the image meets all the criteria, False otherwise.
            plus width, height, channels ; channels will be -1 if not yet computed or not found.
    """
    # Check if the image is loaded properly
    if image is None:
        return False, 0, 0, -1
    # Get image size (width, height)
    width, height = image.size
    # Check for minimum width and height
    if width < min_width or height < min_height:
        return False, width, height, -1
    
    # Determine the number of channels based on image mode
    mode_to_channels = {
        "1": 1,      # (1-bit pixels, black and white)
        "L": 1,      # (8-bit pixels, black and white)
        "P": 1,      # (8-bit pixels, mapped to any other mode using a color palette)
        "RGB": 3,    # (3x8-bit pixels, true color)
        "RGBA": 4,   # (4x8-bit pixels, true color with transparency mask)
        "CMYK": 4,   # (4x8-bit pixels, color separation)
        "YCbCr": 3,  # (3x8-bit pixels, color video format)
        "I": 1,      # (32-bit signed integer pixels)
        "F": 1       # (32-bit floating point pixels)
    }
    # Get the number of channels for the image's mode
    channels = mode_to_channels.get(image.mode, None)
    
    # If the mode is unrecognized, consider the image invalid
    if channels is None:
        return False, width, height, -1
    # Check if the number of channels matches the expected depth
    if channels != depth:
        return False, width, height, channels
    # If all checks pass, the image is valid
    return True, width, height, channels
        

def l2_normalize(embedding: torch.Tensor) -> torch.Tensor:
    """
    embedding: a pytorch tensor.
    return: normalized tensor (1D or 2D).
    """
    norm = embedding.norm(p=2, dim=-1, keepdim=True)
    
    # Check if the norm is zero (i.e., all elements are zero)
    if torch.any(norm == 0):
        print("Warn: zero embeddings cannot be normalized", file=sys.stderr)
        return embedding  # unchanged if all zeros
    
    return embedding / norm


def most_different_vectors(vector_list, k=3, threshold=0.05):
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
    list of indices to selected vectors in given vector_list.
    (empty lists returned if input is a zero vector since that cannot be normalized.)
    """
    n_vectors = len(vector_list)
    
    # Convert list of vectors to 2D numpy array
    vectors = np.array([np.array(v) for v in vector_list])
    
    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if norms.all() == 0.0:
        print(f"zero vector found: {vector_list}", file=sys.stderr)
        return [], []
    normalized_vectors = vectors / norms
    
    # start with a random vector
    selected = [np.random.randint(n_vectors)]
    
    while len(selected) < k:
        # Compute cosine similarities between selected and all vectors
        similarities = np.dot(normalized_vectors, normalized_vectors[selected].T)
        
        # Find the maximum similarity for each vector to any selected vector
        max_similarities = np.max(similarities, axis=1)
        
        # Mask out already selected indices
        max_similarities[selected] = np.inf

        # Find the vector with the minimum max similarity (i.e., most diverse)
        candidate_idx = np.argmin(max_similarities)
        
        # Check if the candidate is diverse enough
        if 1 - max_similarities[candidate_idx] <= threshold:
            # If not diverse enough, stop selecting
            break
        selected.append(candidate_idx)
        
    results = [vector_list[i] for i in selected]        
    return results, selected


def process_input_files(model_path, k=3, neighborhood_threshold=0.08, blip2_model_path):
    """
    a. Read 1 line at a time from stdin  with 1 json per line ("jsonl" format)
    b. extract GIF from the json (base64 encoded) 
    c. get the associated hash. 
    d. Then explode the gif into individual images
    e. process images by OpenCLIP to get embeddings
    f. find the most different <=k embeddings
    g. optionally process the selected images with BLIP2 to get a caption and use that as keywords
    h. emit 1 line of json for each selected embedding the gif hash key (and more if BLIP2 is enabled)
    """
    enable_batching = True
    enable_fp16 = True
    clip_batch_size = 100

    #
    # Initialize OpenCLIP model for image->embedding
    #
    start_model_loading = time.time()
    print("Initialize OpenCLIP model with pretraing dir={model_path}", file=sys.stderr)
    model_file_path = os.path.join(model_path, "openclip_vit_l_14_laion2b_s32b_b82k.pth")
    preprocess_file_path = os.path.join(model_path, "preprocess.pth")
    short_model_name = os.path.basename(model_file_path)
    precision = 'fp16' if enable_fp16 and device == 'cuda' else 'fp32'
    print("model_file_path={model_file_path}", file=sys.stderr)
    print("preprocess_file_path={preprocess_file_path}", file=sys.stderr)

    try:
        # first we create the empty model on device, then load the parameters
        model = open_clip.create_model('ViT-L-14', precision=precision, pretrained=False).to(device)
        # Load the state dict from the local file
        model.load_state_dict(torch.load(model_file_path, map_location=device))
        model.eval()  # Set the model to evaluation mode
        # Load the preprocessing transforms
        preprocess = torch.load(preprocess_file_path)
    except Exception as message:
        print(f"Fatal Exception when trying to load OpenCLIP model: message={message}", file=sys.stderr)
        sys.exit(1) # that's all folks
    
    finish_model_loading = time.time()
    print(f"CLIP2 model loaded in {finish_model_loading-start_model_loading:.2f} sec", file=sys.stderr)
    print(f"clip2_model_name={short_model_name}  k={k}  neighborhood_threshold={neighborhood_threshold}", file=sys.stderr)

    #
    #  Initialize BLIP2 for image->text
    #
    blip = img2txt_blip2.BLIP2Wrapper()
    if not skip_img2txt:
        extractor = keyword_extractor.KeywordExtractor()
        try: 
            blip.initialize(
                model_name="Salesforce/blip2-opt-2.7b",
                temperature=1.0,
                fp16=enable_fp16,
                search_method="beam",
                num_beams=5,
                top_p=0.9
            )
        except Exception as message:
            print(f"Fatal Exception when trying to BLIP2 load model: message={message}", file=sys.stderr)
            sys.exit(2) # that's all folks
    
    # Prepare for writing output
    vector_count_by_hash = defaultdict(int)
    
    embedding_dimensions = 0
    total_gifs_processed = 0
    total_images_processed = 0
    total_embeddings = 0
    total_embeddings_saved = 0
    start_processing = time.time()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        # Process the line to extract hash and GIF binary
        hash_value, gif_binary = process_jsonl_line(line)

        images = extract_images_from_gif(gif_binary, hash=hash_value)
        if len(images) == 0:
            continue

        valid, w, h, chan = validate_image(images[0])
        if not valid:
            print(f"skip image bad shape: hash: {hash_value} w={w} h={h} c={chan}", file=sys.stderr)
            continue # weird image (like 1 pixel high separator image), skip this gif

        total_gifs_processed += 1
        embeddings = []
        batch_images = []

        if enable_batching:
            # process many images in one call to the model.
            # clip_batch_size number of images at a time, so we don't run out of memory on device
            #   in case some humougous gif is out there (area * nframes)
            total_images_processed += len(images)
            # Preprocess and run the OpenCLIP model in batches of size m
            batch_images = []
            embeddings_list = []
            for i in range(0, len(images), clip_batch_size):  # batches within batches
                # Select a batch of m images
                batch = images[i:i+m]
                # Preprocess the batch of images
                batch_images = [preprocess(image).unsqueeze(0) for image in batch]
                # Concatenate the batch of tensors into a single tensor
                batch_tensor = torch.cat(batch_images, dim=0)
                #
                # Run the OpenCLIP model to get embeddings for the batch
                #
                with torch.no_grad():
                    # Move the batch to the device
                    if enable_fp16:
                        batch_tensor = batch_tensor.to(device).half()
                    else:
                        batch_tensor = batch_tensor.to(device)
                    # Encode the batch of images on the device
                    batch_embeddings = model.encode_image(batch_tensor)
                    batch_embeddings = l2_normalize(batch_embeddings)  # normalize on device
                # Move the resulting embeddings back to CPU and append to embeddings_list
                embeddings_list.append(batch_embeddings.cpu().numpy())
            embeddings = np.concatenate(embeddings_list, axis=0)
        else:  # no batching
            for image in images:
                total_images_processed += 1
                image_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension
                with torch.no_grad():
                    image_embedding = model.encode_image(image_tensor).squeeze(0)  # Remove batch dimension
                    image_embedding = l2_normalize(image_embedding)
                    embeddings.append(image_embedding.cpu().numpy())
        # postcondition: now we have (images) and corresponding embeddings (in numpy)

        total_embeddings += len(embeddings)

        # select representative vectors to reduce the number of embeddings 
        selected_embeddings, indices = most_different_vectors(embeddings, k=k, threshold=neighborhood_threshold)
        # in the event that the embeddings were zero, then we will have zero length indices list and
        #   selected_embeddings list; that means no output will occur. Emit msg.
        if len(selected_embeddings) == 0:
            print(f"For hash={hash_value} image embedding was zero", file=sys.stderr)
            continue
        selected_images = [images[i] for i in indices]

        if not skip_img2txt:
            #  run blip over images of selected embeddings
            #   (selected_images is expected to be <=k)
            captions = blip.image_to_text(selected_images, batch_size=3, max_new_tokens=50)
            # deduplicate, remove high frequency words to get keywords
            keywords = extractor.extract_keywords(" ".join(captions))

        previously_processed.add(hash_value) # remember this to avoid duplicates
        total_embeddings_saved += len(selected_embeddings)
        embedding_dimensions = len(selected_embeddings[0]) #remember this to print out for ref
        if skip_img2txt:
            for embedding in selected_embeddings:
                output = {
                    "hash": hash_value,
                    "embedding": embedding.tolist(),  # Convert tensor to list for JSON
                    "mspec": f"{short_model_name}", # in case we change the model/pretraining
                }
                print(json.dumps(output), file=sys.stdout)
                vector_count_by_hash[hash_value] += 1
        else:
            for embedding in selected_embeddings:
                output = {
                    "hash": hash_value,
                    "embedding": embedding.tolist(),  # Convert tensor to list for JSON
                    "mspec": f"{short_model_name}", # in case we change the model/pretraining
                    "keywords": keywords
                }
                print(json.dumps(output), file=sys.stdout)
                vector_count_by_hash[hash_value] += 1

    finish_processing = time.time()
    total_time = finish_processing-start_processing
    print(f"embedding_dimensions={embedding_dimensions}", file=sys.stderr)
    print(f"Total gifs={total_gifs_processed}  images={total_images_processed}  total_embeddings={total_embeddings}  embeddings_saved={total_embeddings_saved}", file=sys.stderr)
    print(f"total_time_secs={total_time:.2f}", file=sys.stderr)
    if total_gifs_processed > 0:
        print(f"gifs/sec={total_gifs_processed/total_time:.2f}  images/sec={total_images_processed/total_time}:.2f", file=sys.stderr)
        print(f"images per gif={total_images_processed/total_gifs_processed:.2f}  embeddings per gif={total_embeddings_saved/total_gifs_processed:.2f}", file=sys.stderr)


    # Calculate and print histogram
    counts = list(vector_count_by_hash.values())
    max_count = max(counts) if counts else 1
    histogram = [counts.count(i) / len(counts) for i in range(1, max_count + 1)]
    print(f"Histogram of vector counts by hash (normalized): [{', '.join(f'{num:.2f}' for num in histogram)}]", file=sys.stderr)

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process images in GIFs, generate embeddings using OpenCLIP, and output results.")
    parser.add_argument("input_files", nargs="+", help="Input JSONL files containing image data")
    parser.add_argument("--model_path", help="path to OpenCLIP model directory with preprocess info in preprocess.pth")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters/vectors to select (default: 3)")
    parser.add_argument("--neighborhood_threshold", type=float, default=0.08, help="Minimum cosine distance between vectors (default: 0.08)")
    parser.add_argument("--blip2_model_path", help="path to BLIP2 model directory with preprocess info in preprocess.pth")

    args = parser.parse_args()

    process_input_files(os.path.expanduser(args.model_path), args.k, args.neighborhood_threshold)




