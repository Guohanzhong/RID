# Bootstrapped from:
# https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
import os
import re
import math
import random
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
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed,ProjectConfiguration

from diffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict
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

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        class_data_root=None,
        class_num=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values}
    return batch


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str=None, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
    """
    Returns:
        a state dict containing just the attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            attn_processors_state_dict[f"{attn_processor_key}.{parameter_key}"] = parameter

    return attn_processors_state_dict

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def main(_):
    config = FLAGS.config
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id
    os.makedirs(config.logdir, exist_ok=True)
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.train.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        mixed_precision=config.db_info.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="ddpo-pytorch", config=config.to_dict(), init_kwargs={"wandb": {"name": config.run_name}}
        )
        logger.info(f"\n{config}")

    if config.seed is not None:
        set_seed(config.seed)
        torch.manual_seed(config.seed)


    if accelerator.is_main_process:
        logger.info('Start loading the diffuser models and schedule')
    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        config.pretrained.model, subfolder="tokenizer", use_fast=False,revision=config.revision
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        config.pretrained.model, subfolder="tokenizer_2", use_fast=False,revision=config.revision
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        config.pretrained.model, 
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        config.pretrained.model, subfolder="text_encoder_2"
    )
    vae = AutoencoderKL.from_pretrained(
        config.pretrained.model, subfolder="vae",revision=config.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained.model, subfolder="unet",revision=config.revision
    )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        config.pretrained.model, subfolder="text_encoder", revision=config.revision
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        config.pretrained.model, subfolder="text_encoder_2",revision=config.revision
    )
    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=config.revision)
    #### define the networks of pipelines
    pipeline.vae,pipeline.unet,pipeline.tokenizer,pipeline.tokenizer_2 = vae,unet,tokenizer_one,tokenizer_two
    pipeline.text_encoder,pipeline.text_encoder_2   = text_encoder_one,text_encoder_two

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipeline.unet= pipeline.unet.to(accelerator.device, dtype=weight_dtype)
    pipeline.vae = pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder = pipeline.text_encoder.to(accelerator.device, dtype=weight_dtype)
    pipeline.text_encoder_2 = pipeline.text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    
    if config.db_info.use_xformers:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            pipeline.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    if config.use_lora:
        unet_lora_attn_procs = {}
        unet_lora_parameters = []
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

            lora_attn_processor_class = (
                LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
            )
            module = lora_attn_processor_class(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=config.train.lora_rank
            )
            unet_lora_attn_procs[name] = module
            unet_lora_parameters.extend(module.parameters())
        unet.set_attn_processor(unet_lora_attn_procs)
    else:
        pipeline.unet.requires_grad_(False)

    if config.train.train_text_encoder:
        text_lora_parameters_one = LoraLoaderMixin._modify_text_encoder(
            text_encoder_one, dtype=torch.float32, rank=config.train.lora_rank
        )
        text_lora_parameters_two = LoraLoaderMixin._modify_text_encoder(
            text_encoder_two, dtype=torch.float32, rank=config.train.lora_rank
        )
    # disable safety checker
    pipeline.safety_checker = None
    # switch to DDIM scheduler/ it is possible to switch to other solvers
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    if accelerator.is_main_process:
        logger.info('Sucessfully loading the diffuser models and schedule')

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = unet_attn_processors_state_dict(model)
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = text_encoder_lora_state_dict(model)
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                    text_encoder_two_lora_layers_to_save = text_encoder_lora_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            if config.train.train_text_encoder:
                StableDiffusionXLPipeline.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                    text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
                )
            else:
                StableDiffusionXLPipeline.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)

        if config.train.train_text_encoder:
            text_encoder_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder." in k}
            LoraLoaderMixin.load_lora_into_text_encoder(
                text_encoder_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_one_
            )

            text_encoder_2_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder_2." in k}
            LoraLoaderMixin.load_lora_into_text_encoder(
                text_encoder_2_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_two_
            )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    if accelerator.is_main_process:
        logger.info('Sucessfully defining the save and load models')

    if config.db_info.with_prior_preservation:
        class_images_dir = Path(config.db_info.class_data_dir)
        os.makedirs(class_images_dir, exist_ok=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < config.db_info.num_class_images:
            torch_dtype = (
                torch.float16 if accelerator.device.type == "cuda" else torch.float32
            )
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                config.pretrained.model,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=config.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = config.db_info.num_class_images - cur_class_images
            if accelerator.is_main_process:
                logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(config.db_info.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=config.train.sample_batch_size
            )

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"],height=config.train.resolution, width=config.train.resolution).images
                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = (
                        class_images_dir
                        / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    )
                    image.save(image_filename)
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)

    if config.train.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if config.train.train_text_encoder:
            pipeline.text_encoder.gradient_checkpointing_enable()
            pipeline.text_encoder_2.gradient_checkpointing_enable()

    if config.train.scale_lr:
        config.train.learning_rate = (
            config.train.learning_rate
            * config.train.gradient_accumulation_steps
            * config.train.train_batch_size
            * accelerator.num_processes
        )

    ######## ######## ######## ######## Define the optimized parameters ######## ######## ######## ########
    optimizer_cls = torch.optim.AdamW
    if config.use_lora:
        params_to_optimize = (
            itertools.chain(unet_lora_parameters, text_lora_parameters_one, text_lora_parameters_two)
            if config.train.train_text_encoder
            else unet_lora_parameters
        )
    else:
        params_to_optimize = unet.parameters()

    optimizer = optimizer_cls(
        params_to_optimize,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        config.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.train.lr_warmup_steps * config.train.gradient_accumulation_steps,
        num_training_steps=config.train.max_train_steps * config.train.gradient_accumulation_steps,
    )
    ######## ######## ######## ######## Define the optimized parameters v ######## ######## ######## ########
    
    if config.train.train_text_encoder:
        unet, text_encoder_one, text_encoder_two, optimizer,lr_scheduler = accelerator.prepare(
            unet, text_encoder_one, text_encoder_two, optimizer,lr_scheduler
        )
    else:
        unet, optimizer,lr_scheduler= accelerator.prepare(
            unet, optimizer,lr_scheduler
        )


    noise_scheduler = DDPMScheduler.from_config(
        config.pretrained.model, subfolder="scheduler"
    )

    train_dataset = DreamBoothDataset(
        instance_data_root=config.db_info.instance_data_dir,
        class_data_root=config.db_info.class_data_dir if config.db_info.with_prior_preservation else None,
        class_num=config.db_info.num_class_images,
        size=config.train.resolution,
        center_crop=config.train.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, config.db_info.with_prior_preservation),
        num_workers=config.train.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.train.gradient_accumulation_steps
    )
    config.train.num_train_epochs = math.ceil(config.train.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = (
        config.train.train_batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    if accelerator.is_main_process:
        print("***** Running training *****")
        print(f"  total batch size: {total_batch_size}")
        print(f"  Finetune the text encoder: {config.train.train_text_encoder}")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num batches each epoch = {len(train_dataloader)}")
        print(f"  Num Epochs = {config.train.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {config.train.train_batch_size}")
        print(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        print(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {config.train.max_train_steps}")
        print(f" The Instance Prompt = {config.db_info.instance_prompt}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(config.train.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0
    last_save = 0

    tokens_one = tokenize_prompt(tokenizer_one, config.db_info.instance_prompt)
    tokens_two = tokenize_prompt(tokenizer_two, config.db_info.instance_prompt)
    if config.db_info.with_prior_preservation:
        class_tokens_one = tokenize_prompt(tokenizer_one, config.db_info.class_prompt)
        tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
        class_tokens_two = tokenize_prompt(tokenizer_two, config.db_info.class_prompt)
        tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)

    for epoch in range(config.train.num_train_epochs):
        unet.train()
        if config.train.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()

        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space
            latents = vae.encode(
                batch["pixel_values"].to(accelerator.device,dtype=torch.float32)
            ).latent_dist.sample()
            latents = latents.to(dtype=weight_dtype)
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (int(bsz),),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            add_time_ids = list((config.train.resolution, config.train.resolution) + (0, 0) + (config.train.resolution, config.train.resolution))
            add_time_ids = torch.tensor([add_time_ids], device=accelerator.device)
            add_time_ids = add_time_ids.repeat(bsz, 1)
            unet_added_conditions = {"time_ids": add_time_ids}
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                                    text_encoders=[text_encoder_one, text_encoder_two],
                                    tokenizers=None,
                                    prompt=None,
                                    text_input_ids_list=[tokens_one, tokens_two],
                                )
            unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
            prompt_embeds_input = prompt_embeds
            model_pred = unet(
                    noisy_latents, timesteps, prompt_embeds_input, added_cond_kwargs=unet_added_conditions
            ).sample
            
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )
            # Dreambooth will use some instances to train a specific lora model;
            # meanwhile, there are some same classes images 
            if config.db_info.with_prior_preservation:
                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                loss = loss + config.db_info.prior_loss_weight * prior_loss
            else:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            avg_loss = accelerator.gather(loss.repeat(config.train.train_batch_size)).mean()
            train_loss += avg_loss.item() / config.train.gradient_accumulation_steps

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = (
                    itertools.chain(unet.parameters(), text_encoder_one.parameters(), text_encoder_two.parameters())
                    if config.train.train_text_encoder
                    else unet.parameters()
                )
                accelerator.clip_grad_norm_(params_to_clip, config.train.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ########
        ######## ######## ######## ######## ########  Save the model ######## ######## ######## ######## ######## ######## ########
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            if global_step != 0 and global_step % config.train.save_steps == 0 and accelerator.is_main_process:
                accelerator.save_state()
        ######## ######## ######## ######## ########  Save the model v ######## ######## ######## ######## ######## ######## ########
        ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ########

        ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ########
        ######## ######## ######## ######## ########  validate the model ######## ######## ######## ######## ######## ######## ########
            if accelerator.is_main_process:
                if config.db_info.validation_prompt is not None and global_step % config.db_info.validation_epochs == 0:
                    pipeline_sample = StableDiffusionXLPipeline.from_pretrained(
                        config.pretrained.model,
                        vae=vae,
                        text_encoder=text_encoder_one,
                        text_encoder_2=text_encoder_two,
                        unet=unet,
                        revision=config.revision,
                        torch_dtype=weight_dtype,
                    )

                    pipeline_sample.scheduler = DPMSolverMultistepScheduler.from_config(
                        pipeline_sample.scheduler.config,
                    )
                    #generator = torch.Generator(device=accelerator.device).manual_seed(config.seed) if config.seed else None
                    images = [
                        pipeline_sample(config.db_info.validation_prompt).images[0]
                        for _ in range(config.db_info.num_validation_images)
                    ]
                    os.makedirs(config.img_output_dir, exist_ok=True)
                    image_save_pth = os.path.join(config.img_output_dir,str(global_step)+'.PNG')
                    images[0].save(image_save_pth)
                    del pipeline_sample
                    torch.cuda.empty_cache()
        ######## ######## ######## ######## ########  validate the model v ######## ######## ######## ######## ######## ######## ########

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= config.train.max_train_steps:
                break

    accelerator.wait_for_everyone()

    accelerator.end_training()
    if accelerator.sync_gradients:
        logger.info("***** Running generating *****")

if __name__ == "__main__":
    app.run(main)
