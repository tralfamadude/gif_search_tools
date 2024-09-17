import sys
import os
import argparse
from PIL import Image

def explode_gif(gif_path, output_dir, basename):
    with Image.open(gif_path) as im:
        for i in range(im.n_frames):
            im.seek(i)
            rgb_im = im.convert('RGB')
            output_filename = f"{basename}-{i+1:02d}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            rgb_im.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="Explode GIF files into individual images.")
    parser.add_argument("gif_files", nargs="+", help="List of GIF files to process")
    parser.add_argument("-o", "--output", required=True, help="Output directory for exploded images")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for gif_file in args.gif_files:
        basename = os.path.splitext(os.path.basename(gif_file))[0]
        explode_gif(gif_file, args.output, basename)
        print(f"Processed {gif_file}")

if __name__ == "__main__":
    main()
