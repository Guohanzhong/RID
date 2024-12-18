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
from libs.uvit import UViT
from libs.DiT import DiT

from diffusers import AutoPipelineForText2Image, DiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)

# Argument parser setup
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with model and folder paths")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("-f", "--folder_path", type=str, required=True, help="Path to the folder containing images")
    return parser.parse_args()

# Load the model
def load_model(torch_device='cuda', lora_path=None, lora_rank=32):
    sd_path = "stable-diffusion-2-1-base"
    save_path = lora_path

    pipe = StableDiffusionPipeline.from_pretrained(sd_path, revision=None, torch_dtype=torch.float16)
    vae = pipe.vae 
    tokenizer = pipe.tokenizer 
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    scheduler = DPMSolverMultistepScheduler.from_pretrained(sd_path, subfolder="scheduler")

    unet_lora_attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            raise NotImplementedError("name must start with up_blocks, mid_blocks, or down_blocks")

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = (
                LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
            )
        unet_lora_attn_procs[name] = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank
        )

    unet.set_attn_processor(unet_lora_attn_procs)
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)
    unet.load_attn_procs(save_path)
    pipe.load_lora_weights(lora_path)

    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)
    return vae, tokenizer, text_encoder, unet, scheduler, pipe

# Main function
if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Set device and seed
    set_seed(4001223)
    device = "cuda"

    # Get model path and folder path from args
    model_path = args.model_path
    folder_path = args.folder_path

    # Load model
    noise_models = DiT(scale=12/255,)
    state_dict = torch.load(model_path, map_location="cuda")
    noise_models.load_state_dict(state_dict, strict=True)
    noise_models = noise_models.to(device)

    noise_temp = 0

    # Process images in the provided folder path
    for n, img in enumerate(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, img)

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

        clean_img = open_image_safely(img_path)
        clean_img = exif_transpose(clean_img)
        clean_img = image_transforms(clean_img).to(device)
        clean_img = clean_img.unsqueeze(0)

        # Inference
        s = time.time()
        noise_added = noise_models(clean_img)
        e = time.time()
        print(f"cost time: {e-s} for one inference at {device}")
        print(noise_added.mean(), torch.abs(noise_added).mean(), noise_added.max())

        attacked_img = clean_img + noise_added
        attacked_img = (attacked_img / 2 + 0.5).clamp(0, 1)

        # Save output image
        save_file = f"output_folder"
        os.makedirs(save_file, exist_ok=True)
        save_image(attacked_img, os.path.join(save_file, f'{n}.png'))

        # Print noise info
        noise_added = (noise_added * 255 / 12) * 127.5 + 128
        print(noise_added.max(), noise_added.min(), noise_added.shape)
