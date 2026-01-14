import os
import argparse
from PIL import Image
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Apply JPEG compression to all images in a folder.")
    parser.add_argument("-i", "--input_dir", type=str, required=True, 
                        help="Path to the input folder containing images.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, 
                        help="Path to the output folder to save compressed images.")
    parser.add_argument("-q", "--quality", type=int, default=75, 
                        help="JPEG compression quality (1-100, default: 75).")
    return parser.parse_args()

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_files = os.listdir(args.input_dir)
    image_files = [f for f in all_files if is_image_file(f)]
    
    print(f"Found {len(image_files)} images in {args.input_dir}")
    print(f"Processing with JPEG Quality = {args.quality}...")

    for filename in tqdm(image_files, desc="Compressing"):
        input_path = os.path.join(args.input_dir, filename)
        
        name_no_ext = os.path.splitext(filename)[0]
        output_filename = f"{name_no_ext}.jpg"
        output_path = os.path.join(args.output_dir, output_filename)

        try:
            with Image.open(input_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output_path, "JPEG", quality=args.quality)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Done! Processed images saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
