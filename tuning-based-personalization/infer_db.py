# *************************************************************************
# Copyright (2023) ML Group @ RUC
# 
# Copyright (2023) SDE-Drag Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *************************************************************************

import torch
import os
import argparse
import random

import numpy as np

from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler,StableDiffusionPipeline
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
import torch.nn.functional as F
from torchvision.utils import  save_image
from torchvision import transforms
from tqdm.auto import tqdm


def load_model(torch_device='cuda',lora_path=None):
    sd_path = "stable-diffusion-2-1-base"

    pipe = StableDiffusionPipeline.from_pretrained(sd_path,revision=None,torch_dtype=torch.float32)
    #vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
    vae  = pipe.vae 
    tokenizer  = pipe.tokenizer 
    #tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(lora_path, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(lora_path, subfolder="unet")
    pipe.unet = unet
    pipe.text_encoder = text_encoder
    #text_encoder = pipe.text_encoder
    #unet = pipe.unet
    scheduler = DPMSolverMultistepScheduler.from_pretrained(sd_path, subfolder="scheduler")

    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)
    pipe.vae = vae
    #pipe.text_encoder = vae
    pipe.unet = unet
    return pipe


@torch.no_grad()
def get_text_embed(prompt: list, tokenizer, text_encoder, torch_device='cuda'):
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    return text_embeddings


@torch.no_grad()
def get_img_latent(img_path, vae, torch_device='cuda', dtype=torch.float32, height=None, weight=None):
    data = Image.open(img_path).convert('RGB')
    if height is not None:
        data = data.resize((weight, height))
    transform = transforms.ToTensor()
    data = transform(data).unsqueeze(0)
    data = (data * 2.) - 1.
    data = data.to(torch_device)
    data = data.to(dtype)
    latents = vae.encode(data).latent_dist.sample()
    latents = 0.18215 * latents
    return latents


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Sampler():
    def __init__(self, model, scheduler, num_steps=100):
        scheduler.set_timesteps(num_steps)
        self.num_inference_steps = num_steps
        self.num_train_timesteps = len(scheduler)

        self.alphas = scheduler.alphas
        self.alphas_cumprod = scheduler.alphas_cumprod

        self.final_alpha_cumprod = torch.tensor(1.0)
        self.initial_alpha_cumprod = torch.tensor(1.0)

        self.model = model

    @torch.no_grad()
    def get_eps(self, img, timestep, guidance_scale, text_embeddings, lora_scale=None):
        latent_model_input = torch.cat([img] * 2) if guidance_scale > 1. else img
        text_embeddings = text_embeddings if guidance_scale > 1. else text_embeddings.chunk(2)[1]

        cross_attention_kwargs = None if lora_scale is None else {"scale": lora_scale}
        with torch.no_grad():
            noise_pred = self.model(latent_model_input, timestep, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample

        if guidance_scale > 1.:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        elif guidance_scale == 1.:
            noise_pred_text = noise_pred
            noise_pred_uncond = 0.
        else:
            raise NotImplementedError(guidance_scale)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        return noise_pred


    def sample(self, timestep, sample, guidance_scale, text_embeddings, sde=False, noise=None, eta=1., lora_scale=None):
        eps = self.get_eps(sample, timestep, guidance_scale, text_embeddings, lora_scale)

        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        sigma_t = eta * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** (0.5) * (1 - alpha_prod_t / alpha_prod_t_prev) ** (0.5) if sde else 0

        pred_original_sample = (sample - beta_prod_t ** (0.5) * eps) / alpha_prod_t ** (0.5)
        pred_sample_direction_coeff = (1 - alpha_prod_t_prev - sigma_t ** 2) ** (0.5)

        noise = torch.randn_like(sample, device=sample.device) if noise is None else noise
        img = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction_coeff * eps + sigma_t * noise

        return img


    def forward_sde(self, timestep, sample, guidance_scale, text_embeddings, eta=1., lora_scale=None):
        prev_timestep = timestep + self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep] if timestep >= 0 else self.initial_alpha_cumprod
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]

        beta_prod_t_prev = 1 - alpha_prod_t_prev

        x_prev = (alpha_prod_t_prev / alpha_prod_t) ** (0.5) * sample + (1 - alpha_prod_t_prev / alpha_prod_t) ** (0.5) * torch.randn_like(sample, device=sample.device)
        eps = self.get_eps(x_prev, prev_timestep, guidance_scale, text_embeddings, lora_scale)

        sigma_t_prev = eta * ((1 - alpha_prod_t) / (1 - alpha_prod_t_prev)) ** (0.5) * (1 - alpha_prod_t_prev / alpha_prod_t) ** (0.5)

        pred_original_sample = (x_prev - beta_prod_t_prev ** (0.5) * eps) / alpha_prod_t_prev ** (0.5)
        pred_sample_direction_coeff = (1 - alpha_prod_t - sigma_t_prev ** 2) ** (0.5)

        noise = (sample - alpha_prod_t ** (0.5) * pred_original_sample - pred_sample_direction_coeff * eps) / sigma_t_prev

        return x_prev, noise


    def forward_ode(self, timestep, sample, guidance_scale, text_embeddings, lora_scale=None):
        prev_timestep = timestep + self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep] if timestep >= 0 else self.initial_alpha_cumprod
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]

        beta_prod_t = 1 - alpha_prod_t

        eps = self.get_eps(sample, timestep, guidance_scale, text_embeddings, lora_scale)
        pred_original_sample = (sample - beta_prod_t ** (0.5) * eps) / alpha_prod_t ** (0.5)
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * eps

        img = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--seed",
            type=int,
            default=12324,
            help='random seed'
    )
    parser.add_argument(
            "--steps",
            type=int,
            default=100,
            help="number of sampling steps"
        )
    parser.add_argument(
            "--scale",
            type=float,
            default=5,
            help="classifier-free guidance scale"
        )
    parser.add_argument(
            "--float64",
            action='store_true',
            help="use double precision"
    )
    opt = parser.parse_args()
    #set_seed(opt.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    lora_path = ""
    pipeline_sample = load_model(device,lora_path)

    pipeline_sample.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline_sample.scheduler.config,
    )
    pipeline_sample.safety_checker = None
    generator = torch.Generator(device=device).manual_seed(423) 
    prompt = "photo of a sks person"
    #prompt = "a dslr photo of sks person,sks person"
    #prompt = "a close photo of sks person, with dslr portrait style"
    #prompt = "a dslr portrait of sks person,photo of sks person"
    prompt = "sks person with a dog"
    images = pipeline_sample(prompt=prompt,guidance_scale=10).images[0]
    images.save("test.png")



if __name__ == "__main__":
    main()
