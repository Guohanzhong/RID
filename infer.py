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

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with model and folder paths")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("-f", "--folder_path", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument("-o", "--output_dir", type=str, default="output_folder", help="Directory to save output images")
    parser.add_argument("--eps", type=float, default=12.0, help="Perturbation budget (e.g. 12 for 12/255)")
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

if __name__ == '__main__':
    args = parse_args()

    set_seed(4001223)
    device = "cuda"

    model_path = args.model_path
    folder_path = args.folder_path
    output_dir = args.output_dir
    scale = args.eps / 255.0

    noise_models = DiT(scale=scale)
    state_dict = torch.load(model_path, map_location="cuda")
    noise_models.load_state_dict(state_dict, strict=True)
    noise_models = noise_models.to(device)
    noise_models.eval()

    os.makedirs(output_dir, exist_ok=True)

    for n, img in enumerate(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, img)
        
        if not img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        size = 512
        center_crop = True
        image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        try:
            clean_img = open_image_safely(img_path)
            clean_img = exif_transpose(clean_img)
            clean_img = image_transforms(clean_img).to(device)
            clean_img = clean_img.unsqueeze(0)

            with torch.no_grad():
                s = time.time()
                noise_added = noise_models(clean_img)
                e = time.time()
                print(f"cost time: {e-s:.4f}s for one inference at {device}")
                print(f"Mean: {noise_added.mean():.4f}, Abs Mean: {torch.abs(noise_added).mean():.4f}, Max: {noise_added.max():.4f}")

                attacked_img = clean_img + noise_added
                attacked_img = (attacked_img / 2 + 0.5).clamp(0, 1)

            save_image(attacked_img, os.path.join(output_dir, f'{n}.png'))

            noise_added_display = (noise_added * 255 / 12) * 127.5 + 128
            print(f"Display Noise - Max: {noise_added_display.max():.4f}, Min: {noise_added_display.min():.4f}, Shape: {noise_added_display.shape}")

        except Exception as e:
            print(f"Error processing {img}: {e}")
