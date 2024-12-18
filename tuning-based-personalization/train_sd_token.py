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
        instance_prompt=None,
        class_prompt=None,
        tokenizer=None
    ):
        self.size = size
        self.center_crop = center_crop

        self.instance_data_root = Path(instance_data_root)
        print(self.instance_data_root)
        print(f"the instance file is {self.instance_data_root}")
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
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.tokenizer = tokenizer

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        ## encode the prompt embeds
        example["instance_prompt_ids"] = self.tokenizer(
        self.instance_prompt,
        padding="do_not_pad",
        truncation=True,
        max_length=self.tokenizer.model_max_length,
        ).input_ids   
        if self.class_data_root:
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

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


def collate_fn(examples,with_prior_preservation=True,tokenizer=None):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = tokenizer.pad(
        {"input_ids": input_ids},
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
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

def save_new_embed(text_encoder, modifier_token_id,modifier_token_list, accelerator, config, output_dir):
    """Saves the new token embeddings from the text encoder."""
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
    for x, y in zip(modifier_token_id, modifier_token_list):
        learned_embeds_dict = {}
        learned_embeds_dict[y] = learned_embeds[x]
        path_pt_token = f"{output_dir}/{y}.bin"
        torch.save(learned_embeds_dict, path_pt_token)
    return path_pt_token


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
    #num_train_timesteps = int(config.sample.num_steps)
    accelerator = Accelerator(
        mixed_precision=config.db_info.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
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

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        config.pretrained.model, 
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
    text_encoder = text_encoder_one

    # Adding a modifier token which is optimized ####
    # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    modifier_token_id = []
    initializer_token_id = []
    if config.db_info.modifier_token is not None:
        modifier_token_list    = config.db_info.modifier_token.split("+")
        initializer_token_list = config.db_info.initializer_token.split("+")
        if len(modifier_token_list) > len(initializer_token_list):
            raise ValueError("You must specify + separated initializer token for each modifier token.")
        for modifier_token, initializer_token in zip(
            modifier_token_list, initializer_token_list[: len(modifier_token_list)]
        ):
            # Add the placeholder token in tokenizer
            num_added_tokens = tokenizer_one.add_tokens(modifier_token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )

            # Convert the initializer_token, placeholder_token to ids
            token_ids = tokenizer_one.encode([initializer_token], add_special_tokens=False)
            print(f'the token ids for the initializer token is {token_ids}')
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id.append(token_ids[0])
            modifier_token_id.append(tokenizer_one.convert_tokens_to_ids(modifier_token))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder_one.resize_token_embeddings(len(tokenizer_one))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder_one.get_input_embeddings().weight.data
        for x, y in zip(modifier_token_id, initializer_token_id):
            token_embeds[x] = token_embeds[y]
            print(f'x and y is {x} and {y}')

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder_one.text_model.encoder.parameters(),
            text_encoder_one.text_model.final_layer_norm.parameters(),
            text_encoder_one.text_model.embeddings.position_embedding.parameters(),
        )
        def freeze_params(params):
            for param in params:
                param.requires_grad = False
        freeze_params(params_to_freeze)

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    if config.db_info.modifier_token is None:
        text_encoder_one.requires_grad_(False)
    unet.requires_grad_(False)

    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=config.revision)
    #### define the networks of pipelines
    pipeline.vae,pipeline.unet,pipeline.tokenizer,pipeline.text_encoder = vae,unet,tokenizer_one,text_encoder_one

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipeline.unet= pipeline.unet.to(accelerator.device, dtype=weight_dtype)
    pipeline.vae = pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder = pipeline.text_encoder.to(accelerator.device, dtype=weight_dtype)
    
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


    if config.train.train_text_encoder:
        text_lora_parameters_one = LoraLoaderMixin._modify_text_encoder(
            text_encoder_one, dtype=torch.float32, rank=config.train.lora_rank
        )
    #if accelerator.is_main_process:
    #    for _up, _down in extract_lora_ups_down(unet):
    #        print("Before training: Unet First Layer lora up", _up.weight.data)
    #        print("Before training: Unet First Layer lora down", _down.weight.data)
    #        break

    # disable safety checker
    pipeline.safety_checker = None
    # switch to DDIM scheduler/ it is possible to switch to other solvers
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    if accelerator.is_main_process:
        logger.info('Sucessfully loading the diffuser models and schedule')

    if config.db_info.with_prior_preservation:
        class_images_dir = Path(config.db_info.class_data_dir)
        os.makedirs(class_images_dir, exist_ok=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < config.db_info.num_class_images:
            torch_dtype = (
                torch.float16 if accelerator.device.type == "cuda" else torch.float32
            )
            pipeline = StableDiffusionPipeline.from_pretrained(
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
        os.makedirs(config.img_output_dir, exist_ok=True)

    if config.train.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if config.train.train_text_encoder or config.db_info.modifier_token is not None:
            pipeline.text_encoder.gradient_checkpointing_enable()

    if config.train.scale_lr:
        config.train.learning_rate = (
            config.train.learning_rate
            * config.train.gradient_accumulation_steps
            * config.train.train_batch_size
            * accelerator.num_processes
        )
    print(f'the training learning rate is {config.train.learning_rate}')
    ######## ######## ######## ######## Define the optimized parameters ######## ######## ######## ########
    optimizer_cls = torch.optim.AdamW
    if config.use_lora:
        if config.db_info.modifier_token is not None:
            params_to_optimize = (
                [
                    {"params": itertools.chain(*text_encoder_one.get_input_embeddings().parameters()),"lr": config.train.learning_rate,},
                    {"params": itertools.chain(*text_encoder_lora_params),"lr": config.train.learning_rate,},
                ]
                if config.train.train_text_encoder
                else text_encoder_one.get_input_embeddings().parameters()
            )
        else:
            params_to_optimize = (
                [
                    {"params": itertools.chain(unet_lora_params.parameters()), "lr": config.train.learning_rate},
                    {"params": itertools.chain(*text_encoder_lora_params),"lr": config.train.learning_rate,},
                ]
                if config.train.train_text_encoder
                else itertools.chain(unet_lora_params.parameters())
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
    
    if config.train.train_text_encoder or config.db_info.modifier_token is not None:
        unet, text_encoder_one, optimizer,lr_scheduler = accelerator.prepare(
            unet, text_encoder_one, optimizer,lr_scheduler
        )
    else:
        unet, optimizer,lr_scheduler= accelerator.prepare(
            unet, optimizer,lr_scheduler
        )


    noise_scheduler = DDPMScheduler.from_config(
        config.pretrained.model, subfolder="scheduler"
    )

    train_dataset = DreamBoothDataset(
        instance_prompt=config.db_info.instance_prompt,
        instance_data_root=config.db_info.instance_data_dir,
        class_prompt=config.db_info.class_prompt,
        class_data_root=config.db_info.class_data_dir if config.db_info.with_prior_preservation else None,
        class_num=config.db_info.num_class_images,
        size=config.train.resolution,
        center_crop=config.train.center_crop,
        tokenizer=tokenizer_one,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, config.db_info.with_prior_preservation,tokenizer=tokenizer_one),
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
        print(f" The Modifier Token = {config.db_info.modifier_token}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(config.train.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0
    last_save = 0

    for epoch in range(config.train.num_train_epochs):
        #unet.train()
        if config.train.train_text_encoder or config.db_info.modifier_token is not None:
            text_encoder_one.train()
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
            #timesteps = torch.cat([timesteps,timesteps],dim=0)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder_one(batch["input_ids"].to(accelerator.device))[0]
            
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            target = noise
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

            # embeddings for the concept, as we only want to optimize the concept embeddings
            if config.db_info.modifier_token is not None:
                #print(text_encoder_one.get_input_embeddings().weight.grad)
                if accelerator.num_processes > 1:
                    grads_text_encoder = text_encoder_one.module.get_input_embeddings().weight.grad
                else:
                    grads_text_encoder = text_encoder_one.get_input_embeddings().weight.grad
                # Get the index for tokens that we want to zero the grads for
                index_grads_to_zero = torch.arange(len(tokenizer_one)) != modifier_token_id[0]
                for i in range(len(modifier_token_id[1:])):
                    index_grads_to_zero = index_grads_to_zero & (
                        torch.arange(len(tokenizer_one)) != modifier_token_id[i]
                    )
                grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[
                    index_grads_to_zero, :
                ].fill_(0)
        
            if accelerator.sync_gradients:
                if (config.db_info.modifier_token is not None) or (config.train.train_text_encoder):
                    params_to_clip = ( 
                        itertools.chain(unet.float().parameters(), text_encoder_one.float().parameters())
                        )
                else:
                    params_to_clip = ( unet.float().parameters() )
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
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if config.train.num_checkpoint_limit is not None:
                            os.makedirs(config.output_dir, exist_ok=True)
                            checkpoints = os.listdir(config.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.train.num_checkpoint_limit:
                                num_to_remove = len(checkpoints) - config.train.num_checkpoint_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(config.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        filename_safe = f"{save_path}/lora_weight.safetensors"
                        os.makedirs(save_path, exist_ok=True)
                        #loras = {}
                        #loras["unet"] = (unet, {"CrossAttention", "Attention", "GEGLU"})
                        #save_safeloras(loras, filename_safe)
                        #pt_token_path = f"{save_path}/token.pt"
                        #LoraLoaderMixin.save_lora_weights(
                        #    save_directory=save_path,
                        #    unet_lora_layers=unet_lora_params,
                        #    text_encoder_lora_layers=None,
                        #)
                        if config.db_info.modifier_token is not None:
                            token_savepath = save_new_embed(text_encoder_one, modifier_token_id,modifier_token_list, accelerator, config, save_path)
                        #unet.save_attn_procs(save_path)
                        #accelerator.save_state(save_path)
                        #logger.info(f"Saved unet-state to {save_path}")

            ######## ######## ######## ######## ########  Save the model v ######## ######## ######## ######## ######## ######## ########
            ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ########

            ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ######## ########
            ######## ######## ######## ######## ########  validate the model ######## ######## ######## ######## ######## ######## ########
            if accelerator.is_main_process:
                if config.db_info.validation_prompt is not None and global_step % config.db_info.validation_epochs == 0:
                    pipeline_sample = StableDiffusionPipeline.from_pretrained(
                        config.pretrained.model,
                        vae=vae,
                        text_encoder=text_encoder_one,
                        unet=unet,
                        revision=config.revision,
                        torch_dtype=weight_dtype,
                    )
                    
                    #unet, text_encoder_one = accelerator.prepare(unet,text_encoder_one)
                    #try:
                    #    pipeline_sample.unet.load_attn_procs(save_path)
                    #    logger.info(f"loading unet-state from {save_path}")
                    #except:
                    #    pipeline_sample.unet = unet
                    pipeline_sample.unet.eval()
                    if config.db_info.modifier_token is not None:
                        print(token_savepath)
                        pipeline_sample.load_textual_inversion(token_savepath)
                    pipeline_sample = pipeline_sample.to(accelerator.device)

                    pipeline_sample.scheduler = DPMSolverMultistepScheduler.from_config(
                        pipeline_sample.scheduler.config,
                    )
                    pipeline_sample.safety_checker = None
                    generator = torch.Generator(device=accelerator.device).manual_seed(config.seed+global_step) if config.seed else None
                    images = [
                        pipeline_sample(config.db_info.validation_prompt,generator=generator).images[0]
                        for _ in range(config.db_info.num_validation_images)
                    ]
                    os.makedirs(config.img_output_dir, exist_ok=True)
                    image_save_pth = os.path.join(config.img_output_dir,str(global_step)+'.PNG')
                    logger.info(f'image saved in {image_save_pth}')
                    images[0].save(image_save_pth)

                    sample2= True
                    if sample2:
                        images = [
                            pipeline_sample("photo of a <new1> person",generator=generator).images[0]
                            for _ in range(config.db_info.num_validation_images)
                        ]
                        os.makedirs(config.img_output_dir, exist_ok=True)
                        image_save_pth = os.path.join(config.img_output_dir,str(global_step)+'_test.PNG')
                        logger.info(f'image saved in {image_save_pth}')
                        images[0].save(image_save_pth)

                    del pipeline_sample
                    #torch.cuda.empty_cache()
                ####### ######## ######## ######## ########  validate the model v ######## ######## ######## ######## ######## ######## ########
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
