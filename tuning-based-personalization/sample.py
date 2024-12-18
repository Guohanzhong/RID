import torch
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
    DDIMScheduler
)

torch_dtype = torch.float32
pipeline = StableDiffusionPipeline.from_pretrained(
                "stable-diffusion-2-1",
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=None,
            )
pipeline = pipeline.to('cuda:1')
pipeline.set_progress_bar_config(disable=True)
resolution = 512
images = pipeline('a dog and a cat standing together',height=resolution, width=resolution,num_inference_steps=100,guidance_scale=7).images
#print(images)
images[0].save('test.png')