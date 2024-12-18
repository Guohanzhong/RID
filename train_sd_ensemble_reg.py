import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
from typing import List
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from packaging import version
from tqdm import trange
from safetensors import safe_open

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel,StableDiffusionXLPipeline,DPMSolverMultistepScheduler,StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available

from transformers import CLIPImageProcessor
from transformers import CLIPModel,CLIPProcessor
from transformers import AutoTokenizer, PretrainedConfig
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from torch.cuda.amp import custom_bwd, custom_fwd

from libs.uvit import UViT
from libs.DiT import DiT

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        (gt_grad,) = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def load_model(args, model_path,accelerator):
    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(model_path, None)

    # Load scheduler and models
    text_encoder = text_encoder_cls.from_pretrained(
        model_path,
        subfolder="text_encoder",
        revision=None,
    )
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", revision=None)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", revision=None)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    return text_encoder, unet, tokenizer, noise_scheduler, vae

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def open_image_safely(image_path):
    from PIL import PngImagePlugin
    LARGE_ENOUGH_NUMBER = 100
    PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
    from PIL import Image    
    img = Image.open(image_path).convert("RGBA")
    background = Image.new("RGBA", img.size, "white")
    background.paste(img, (0, 0), img)  # Image.paste(im, box, mask)
    if not background.mode == "RGB":
        background = background.convert("RGB")
    img = background
    return img

logger = get_logger(__name__)
# Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, json_file,tokenizer,size=512,center_crop=True,pair_json=None,):
        super().__init__()
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png"}]
        self.pair_data = json.load(open(pair_json)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        
    def __getitem__(self, idx):
        item = self.data[idx] 
        image_file = item["image_file"]

        pair_item = self.pair_data[idx % len(self.pair_data)] 

        source_file = pair_item["source_path"]
        target_file = pair_item["attacked_path"]
        
        # read image safely
        #raw_image = Image.open(image_file)
        raw_image = open_image_safely(image_file)

        source_image = open_image_safely(source_file)
        target_image = open_image_safely(target_file)

        image_tensor = self.transform(raw_image.convert("RGB"))
        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])
        
        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )

        image_tensor_source = self.transform(source_image.convert("RGB"))
        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])
        
        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        source_image = transforms.functional.crop(
            image_tensor_source, top=top, left=left, height=self.size, width=self.size
        )
        image_tensor_target = self.transform(target_image.convert("RGB"))
        target_image = transforms.functional.crop(
            image_tensor_target, top=top, left=left, height=self.size, width=self.size
        )

        text_input_ids = self.tokenizer(
            " ",
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "source_image":source_image,
            "target_image":target_image,
        }
        
    
    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    source_images = torch.stack([example["source_image"] for example in data])
    target_images = torch.stack([example["target_image"] for example in data])

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "source_images":source_images,
        "target_images":target_images
    }
    

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--seed",
        type=int,
        default=401234,
        help="the seed for experiments",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stable-diffusion-2-1-base,stable-diffusion-v1-5,stable-diffusion-v1-4",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_noise_model",
        type=str,
        default=None,
        help="Path to pretrained model of noise network.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        help="Training data",
    )
    parser.add_argument(
        "--pair_path",
        type=str,
        default=None,
        help="Training pairs data for regularization loss",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./train_cache/pth/test",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--tensorboard_output_dir",
        type=str,
        default="./train_cache/pth/test",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--vad_output_dir",
        type=str,
        default="./train_cache/image/test",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    ####### the model config for ip-adapter
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--eps_scale",
        type=float,
        default=6/255,
        help="Constraint level for adding noise.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument(
        "--train_batch_size", type=int, default=3, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if accelerator.is_main_process:
        writer = SummaryWriter(args.tensorboard_output_dir)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.

    model_paths = list(args.pretrained_model_name_or_path.split(","))
    num_models = len(model_paths)

    # MODEL_NAMES = ["text_encoder", "unet", "tokenizer", "noise_scheduler", "vae"]
    MODEL_BANKS = [load_model(args, path, accelerator) for path in model_paths]
    MODEL_STATEDICTS = [
        {
            "text_encoder": MODEL_BANKS[i][0].state_dict(),
            "unet": MODEL_BANKS[i][1].state_dict(),
        }
        for i in range(num_models)
    ]


    ## load the pretrained-models
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #pipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, revision=None)
    noise_scheduler_1 = DDPMScheduler.from_pretrained(model_paths[0], subfolder="scheduler")
    tokenizer_1 = CLIPTokenizer.from_pretrained(model_paths[0], subfolder="tokenizer")
    text_encoder_1 = CLIPTextModel.from_pretrained(model_paths[0], subfolder="text_encoder")
    vae_1 = AutoencoderKL.from_pretrained(model_paths[0], subfolder="vae")
    unet_1 = UNet2DConditionModel.from_pretrained(model_paths[0], subfolder="unet")
    alphas_1 = noise_scheduler_1.alphas_cumprod.to(accelerator.device)  # for convenience
    unet_1.to(accelerator.device, dtype=weight_dtype)
    vae_1.to(accelerator.device) # use fp32
    text_encoder_1.to(accelerator.device, dtype=weight_dtype)

    unet_1.requires_grad_(False)
    vae_1.requires_grad_(False)
    text_encoder_1.requires_grad_(False)

    noise_scheduler_2 = DDPMScheduler.from_pretrained(model_paths[1], subfolder="scheduler")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_paths[1], subfolder="tokenizer")
    text_encoder_2 = CLIPTextModel.from_pretrained(model_paths[1], subfolder="text_encoder")
    vae_2 = AutoencoderKL.from_pretrained(model_paths[1], subfolder="vae")
    unet_2 = UNet2DConditionModel.from_pretrained(model_paths[1], subfolder="unet")
    alphas_2 = noise_scheduler_2.alphas_cumprod.to(accelerator.device)  # for convenience
    unet_2.to(accelerator.device, dtype=weight_dtype)
    vae_2.to(accelerator.device) # use fp32
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    unet_2.requires_grad_(False)
    vae_2.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    noise_scheduler_3 = DDPMScheduler.from_pretrained(model_paths[2], subfolder="scheduler")
    tokenizer_3 = CLIPTokenizer.from_pretrained(model_paths[2], subfolder="tokenizer")
    text_encoder_3 = CLIPTextModel.from_pretrained(model_paths[2], subfolder="text_encoder")
    vae_3 = AutoencoderKL.from_pretrained(model_paths[2], subfolder="vae")
    unet_3 = UNet2DConditionModel.from_pretrained(model_paths[2], subfolder="unet")
    alphas_3 = noise_scheduler_3.alphas_cumprod.to(accelerator.device)  # for convenience
    unet_3.to(accelerator.device, dtype=weight_dtype)
    vae_3.to(accelerator.device) # use fp32
    text_encoder_3.to(accelerator.device, dtype=weight_dtype)

    unet_3.requires_grad_(False)
    vae_3.requires_grad_(False)
    text_encoder_3.requires_grad_(False)

    ## loaded the pretrained-models


    #noise_models = UViT()
    noise_models = DiT(scale=args.eps_scale)
    noise_models.to(accelerator.device)

    if args.pretrained_noise_model is not None:
        state_dict = torch.load(args.pretrained_noise_model, map_location="cpu")
        noise_models.load_state_dict(state_dict, strict=True)
    params_to_opt = itertools.chain(noise_models.parameters())


    # optimizer
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    tokenizer = CLIPTokenizer.from_pretrained(model_paths[0], subfolder="tokenizer")
    train_dataset = MyDataset(args.data_json_file,tokenizer=tokenizer_1,size=args.resolution,pair_json=args.pair_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Prepare everything with our `accelerator`.
    noise_models, optimizer, train_dataloader = accelerator.prepare(noise_models, optimizer, train_dataloader)
    
    global_step = 0
    noise_models.train()
    for epoch in trange(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(noise_models):
                # Convert images to latent space
                loss_ = []
                loss_sds = []
                image_ = batch["images"].to(accelerator.device, dtype=torch.float32)
                image_s = batch["source_images"].to(accelerator.device, dtype=torch.float32)

                infer_images = torch.cat([image_,image_s],dim=0)
                added_noise_all = noise_models(infer_images)
                #added_noise_all = torch.clamp(added_noise_all.data, -10/255, 10/255)
                added_noise,added_noise_s = torch.chunk(added_noise_all, 2, dim=0)

                #added_noise = noise_models(image_)
                image_ = image_ + added_noise

                #added_noise_s = noise_models(image_s)
                #image_s = image_s + added_noise_s
                image_t = batch["target_images"].to(accelerator.device, dtype=torch.float32)
                predict_noise = (image_t - image_s).detach()
                loss_r =  10*F.mse_loss(added_noise_s.float(), predict_noise.float(), reduction="mean")
                #loss_r = 4*F.l1_loss(added_noise_s.float(), predict_noise.float(), reduction="mean")
                
                loss_.append(loss_r)
#################
                loss_ = torch.stack(loss_).view(1, 1)
                loss = torch.mean(loss_)


                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                #avg_loss_sds = accelerator.gather(loss_sds.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print(f"the max of added noise is {predict_noise.max()}")
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}, reg_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss, loss_r))
                    #writer.add_scalar("loss", avg_loss, global_step=step)
                    writer.add_scalar("loss", avg_loss, global_step=global_step)
                    writer.add_scalar("reg_loss",loss_r, global_step=global_step)
                    #writer.add_scalar("sds_loss",avg_loss_sds, global_step=global_step)
            global_step += 1
            if accelerator.sync_gradients:
                if global_step % args.save_steps == 0 and accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(noise_models)
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path,exist_ok=True)
                    accelerator.save(unwrapped_model.state_dict(), os.path.join(save_path, f"noise_model_modules.bin"))            
            begin = time.perf_counter()
    if accelerator.is_main_process:
        writer.close()
                
if __name__ == "__main__":
    main()    
