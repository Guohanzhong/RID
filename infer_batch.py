import torch
import time
import os
import argparse
import random
import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms
from torchvision.utils import save_image
from libs.DiT import DiT
import multiprocessing as mp
import re
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Run batch inference with multi-GPU")
    parser.add_argument("-m", "--model_path", type=str, required=True,
                       help="Path to the model")
    parser.add_argument("-r", "--root_path", type=str, required=True,
                       help="Root directory containing multiple image folders")
    parser.add_argument("-o", "--output_root", type=str, required=True,
                       help="Root directory for saving processed images")
    parser.add_argument("-n", "--num_gpus", type=int, default=8,
                       help="Number of GPUs to use")
    parser.add_argument("--eps", type=float, default=12.0,
                       help="Perturbation budget (e.g. 12 for 12/255)")
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def open_image_safely(image_path):
    from PIL import PngImagePlugin
    LARGE_ENOUGH_NUMBER = 100
    PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
    from PIL import Image
    img = Image.open(image_path).convert("RGBA")
    background = Image.new("RGBA", img.size, "white")
    background.paste(img, (0, 0), img)
    if not background.mode == "RGB":
        background = background.convert("RGB")
    img = background
    return img

def is_image_file(filename):
    return re.search(r"\.(png|jpg|jpeg)$", filename, re.IGNORECASE)

def process_folder(gpu_id, folders, model_path, output_root, scale):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    model = DiT(scale=scale)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    size = 512
    center_crop = True
    image_transforms = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    for folder in tqdm(folders, desc=f"GPU-{gpu_id}", position=gpu_id):
        folder_name = os.path.basename(folder)
        output_dir = os.path.join(output_root, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        if folder.lower().endswith('.zip'):
            continue

        for img_file in os.listdir(folder):
            if not is_image_file(img_file):
                continue

            img_path = os.path.join(folder, img_file)
            output_path = os.path.join(output_dir, img_file)

            if os.path.exists(output_path):
                continue

            try:
                with torch.no_grad():
                    clean_img = open_image_safely(img_path)
                    clean_img = exif_transpose(clean_img)
                    clean_img = image_transforms(clean_img).to(device)
                    clean_img = clean_img.unsqueeze(0)

                    noise_added = model(clean_img)

                    attacked_img = clean_img + noise_added
                    attacked_img = (attacked_img / 2 + 0.5).clamp(0, 1)
                    save_image(attacked_img[0], output_path)

            except Exception as e:
                print(f"Error processing {img_path} on GPU-{gpu_id}: {str(e)}")

def main():
    args = parse_args()
    set_seed(4001223)

    os.makedirs(args.output_root, exist_ok=True)

    all_items = [os.path.join(args.root_path, d) for d in os.listdir(args.root_path)]
    folders = [f for f in all_items if os.path.isdir(f) and not f.lower().endswith('.zip')]

    folder_chunks = np.array_split(folders, args.num_gpus)
    
    scale = args.eps / 255.0

    processes = []
    for gpu_id in range(args.num_gpus):
        p = mp.Process(target=process_folder,
                      args=(gpu_id, folder_chunks[gpu_id].tolist(),
                            args.model_path, args.output_root, scale))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
