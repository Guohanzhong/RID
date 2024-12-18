import os
import requests
import torch
from PIL import Image
from io import BytesIO
from tqdm import tqdm

from diffusers import StableDiffusionImg2ImgPipeline

if __name__ == "__main__":
    device = "cuda"
    model_id_or_path = "stable-diffusion-2-1-base"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    num_list = ["01","02","03","04","05","06","07","08","09","10","22","28","33","47"]
    for used_name in tqdm(num_list):
    #used_name = "15"
        save_file = f"EXP/Exp/attack/diffpure/{used_name}"
        top_dir = f"Exp/attack/eps12-255-gam/{used_name}"
        os.makedirs(save_file, exist_ok=True)
        top_list = os.listdir(top_dir)
        for ele in top_list:
            image_file = os.path.join(top_dir,ele)
            init_image = Image.open(image_file).convert("RGB")
            init_image = init_image.resize((512, 512))
            prompt = "A photo of a person"
            images = pipe(prompt=prompt, image=init_image, strength=0.1, guidance_scale=7.5).images
            save_image = os.path.join(save_file,ele)
            images[0].save(save_image)