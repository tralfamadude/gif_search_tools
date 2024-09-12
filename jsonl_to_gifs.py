# read jsonl from stdin and extract and write the contained gif files using the hash as the basename. 
import sys
import os
import json
import base64

def process_jsonl_line(jsonl_string):
    # Parse the JSON string
    data = json.loads(jsonl_string)
    
    # Extract the hash
    hash_value = data.get("hash")
    
    # Extract and decode the gifb64 field
    gif_base64 = data.get("gifb64")
    gif_binary = base64.b64decode(gif_base64)
    
    # Return the tuple with the hash and binary content
    return hash_value, gif_binary

def main():
    # Ensure the destination directory is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <destination_directory>")
        sys.exit(1)
    
    # Get the destination directory from the command line
    destination_dir = sys.argv[1]
    
    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    # Read JSONL input from stdin line by line
    for line in sys.stdin:
        # Process each line to get the hash and gif binary
        hash_value, gif_binary = process_jsonl_line(line.strip())
        
        if hash_value:
            # Build the file path using the hash as the basename
            gif_filename = f"{hash_value}.gif"
            gif_filepath = os.path.join(destination_dir, gif_filename)
            
            # Write the binary GIF data to the file
            with open(gif_filepath, "wb") as gif_file:
                gif_file.write(gif_binary)

if __name__ == "__main__":
    main()

