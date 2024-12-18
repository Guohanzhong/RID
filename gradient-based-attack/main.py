
import json
import pandas as pd
import torch.nn as nn
# Bootstrapped from:
# https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
import os
import re
import math
import random
import shutil
import datetime
import argparse
import hashlib
import itertools
import inspect
from absl import app, flags
from packaging import version
from ml_collections import config_flags

from pathlib import Path
from typing import Dict
from datetime import date
from typing import Optional
from PIL.ImageOps import exif_transpose

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import torch.utils.checkpoint


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
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)


from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed,ProjectConfiguration

from diffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict,AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available

from tqdm.auto import tqdm
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer,CLIPTextModelWithProjection
from transformers import AutoTokenizer,PretrainedConfig

from PIL import Image
from torch.utils.data import Dataset

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
logger = get_logger(__name__)

class AlignDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        image_file = "/2d-cfs-nj/alllanguo/code/test/Attack/demo",
        json_file = "/2d-cfs-nj/alllanguo/code/test/assets/face_data.json",
        size=512,
        center_crop=False,
        tokenizer=None,
        choose_iterations = 50
    ):
        self.size = size
        self.center_crop = center_crop

        self.attack_img = []
        self.clean_img  = []
        self.top_dir  = image_file
        self.json_path= json_file
        with open(self.json_path, 'r') as file:  
            data_dict = {}  
            for line in file:  
                row_dict = json.loads(line)  
                data_dict.update(row_dict)  
        file_list = os.listdir(self.top_dir)
        accumalated_number = 0
        for img_ele in file_list:
            file_path = os.path.join(self.top_dir,img_ele,str(choose_iterations))
            file_path_list = os.listdir(file_path)
            for images in file_path_list:
                self.attack_img.append(os.path.join(file_path,images))
                #number_file = int(img_ele) + int(images.split('_')[-1].split('.')[0])
                self.clean_img.append(data_dict[str(accumalated_number)])
                accumalated_number += 1

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.attack_img)

    def __getitem__(self, index):
        example = {}
        ## encode the prompt embeds
        attacked_image = Image.open(self.attack_img[index])
        cleaned_image = Image.open(self.clean_img[index])

        attacked_image = exif_transpose(attacked_image)
        if not attacked_image.mode == "RGB":
            attacked_image = attacked_image.convert("RGB")
        example["attacked_image"] = self.image_transforms(attacked_image)

        cleaned_image = exif_transpose(cleaned_image)
        if not cleaned_image.mode == "RGB":
            cleaned_image = cleaned_image.convert("RGB")
        example["cleaned_image"] = self.image_transforms(cleaned_image)

        example["predicted_noise"] = example["attacked_image"] - example["cleaned_image"]
        return example


def collate_fn(examples):
    pixel_values_poi = [example["attacked_image"] for example in examples]
    pixel_values_cle = [example["cleaned_image"] for example in examples]
    pixel_values_noi = [example["predicted_noise"] for example in examples]

    pixel_values_poi = torch.stack(pixel_values_poi)
    pixel_values_poi = pixel_values_poi.to(memory_format=torch.contiguous_format).float()

    pixel_values_cle = torch.stack(pixel_values_cle)
    pixel_values_cle = pixel_values_cle.to(memory_format=torch.contiguous_format).float()

    pixel_values_noi = torch.stack(pixel_values_noi)
    pixel_values_noi = pixel_values_noi.to(memory_format=torch.contiguous_format).float()


    batch = {
        "pixel_values_attacked": pixel_values_poi,
        "pixel_values_clean": pixel_values_cle,
        "pixel_values_noise": pixel_values_noi,
    }
    return batch

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3

train_dataset = AlignDataset(
    image_file="demo",
    json_file ="face_data.json",
    size=512,
    center_crop=True,
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda examples: collate_fn(examples),
    num_workers=8,
)

# 创建U-Net模型实例
unet = UNet(in_channels=3, out_channels=3)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    for data in train_dataloader:
        inputs, targets = data["pixel_values_clean"],data["pixel_values_noise"]
        optimizer.zero_grad()
        outputs = unet(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f} and output max {outputs.max()}")