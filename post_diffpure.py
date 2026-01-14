import os
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionImg2ImgPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Post-process a folder of images using a diffusion-based method.")
    parser.add_argument("-i", "--input_dir", type=str, required=True, 
                        help="Root directory containing subfolders of images to be processed.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, 
                        help="Directory to save the processed images.")
    parser.add_argument("-m", "--model_path", type=str, required=True, 
                        help="Path to the Stable Diffusion model.")
    parser.add_argument("-p", "--process_list", type=str, 
                        help="Comma-separated list of subfolder names to process. If not provided, all subfolders in the input directory will be processed.")
    parser.add_argument("-s", "--strength", type=float, default=0.1, 
                        help="The strength for img2img transformation, controlling how much the image is altered.")
    parser.add_argument("-g", "--guidance_scale", type=float, default=7.5, 
                        help="Guidance scale for the diffusion model.")
    parser.add_argument("--prompt", type=str, default="A photo of a person", 
                        help="Prompt to guide the image generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run the model on (e.g., 'cuda', 'cpu').")
    return parser.parse_args()

def main():
    args = parse_args()

    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
        pipe = pipe.to(args.device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    if args.process_list:
        sub_dirs = args.process_list.split(',')
    else:
        sub_dirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    
    print(f"Sub-directories to be processed: {sub_dirs}")

    for sub_dir_name in tqdm(sub_dirs, desc="Processing Batches"):
        input_folder = os.path.join(args.input_dir, sub_dir_name)
        output_folder = os.path.join(args.output_dir, sub_dir_name)
        
        if not os.path.isdir(input_folder):
            print(f"Warning: Input directory does not exist, skipping: {input_folder}")
            continue

        os.makedirs(output_folder, exist_ok=True)
        
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        for image_name in tqdm(image_files, desc=f"Processing images in {sub_dir_name}", leave=False):
            image_path = os.path.join(input_folder, image_name)
            
            try:
                init_image = Image.open(image_path).convert("RGB")
                init_image = init_image.resize((512, 512))
                
                images = pipe(
                    prompt=args.prompt, 
                    image=init_image, 
                    strength=args.strength, 
                    guidance_scale=args.guidance_scale
                ).images
                
                save_path = os.path.join(output_folder, image_name)
                images[0].save(save_path)
            except Exception as e:
                print(f"Failed to process image {image_path}: {e}")

if __name__ == "__main__":
    main()
