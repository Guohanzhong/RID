import math
import logging 
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torch.cuda.amp import autocast

import torch
from torch import multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils._pytree import tree_map

from diffusers import (StableDiffusionPipeline, 
                       StableDiffusionInpaintPipeline,
                       EulerAncestralDiscreteScheduler,
                       DPMSolverMultistepScheduler,
                       UniPCMultistepScheduler,
                       KDPM2AncestralDiscreteScheduler,
                       DDIMScheduler)

import torchvision.transforms as T
#from diffusers.utils import randn_tensor

import builtins
import ml_collections
import accelerate
import tempfile

import utils
from utils import (preprocess, 
                    prepare_mask_and_masked_image,
                    prepare_nonmask_and_nonmasked_image, 
                    recover_image, 
                    prepare_image)
from dataset import IMAGE
to_pil = T.ToPILImage()

def poison_loss(unet,autoencoder,pipeline,
                X_adv,cur_mask,mask,text_embeddings,z_shape,**kwargs):
    unet.zero_grad()
    X_adv.requires_grad = True
    cur_mask = cur_mask.clone()
    cur_mask.requires_grad = False

    image_latents = autoencoder.encode(X_adv).latent_dist.sample()
    image_latents = autoencoder.config.scaling_factor * image_latents

    timesteps = torch.randint(0,pipeline.scheduler.config.num_train_timesteps,
        (image_latents.shape[0],),device=image_latents.device,)
    timesteps = timesteps.long()
    
    shape = z_shape
    noise = torch.randn_like(image_latents)
    noise = noise.to(device=pipeline.device, dtype=text_embeddings.dtype)
    #noise = randn_tensor(shape,device=pipeline.device, dtype=text_embeddings.dtype)
    noisy_latents = pipeline.scheduler.add_noise(image_latents, noise, timesteps)  
    ## noisy x_t
    #latent_model_input = torch.cat([image_latents] * 2)
    #with autocast():  
    noise_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=text_embeddings,return_dict=False,
    )[0]
    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

    params_to_grad = []
    for name, param in unet.named_parameters():
        params_to_grad.append(param)
    loss_grad_theta = torch.autograd.grad(loss, params_to_grad, retain_graph=True, create_graph=True)
    grads = 0
    test_grads = 0
    for grad_ele in (loss_grad_theta):
        #grad_ele = grad_ele.reshape(-1,1)
        #for kk in range(grad_ele.shape[0]):
        #    ans_grad = (torch.autograd.grad(grad_ele[0,:].pow(2).sum(), X_adv,retain_graph=True)[0]).max()
        #    if torch.isnan(ans_grad) == torch.tensor(True).to(grad_ele.device):
        #        continue
        #    else:
        #        grads += ans_grad
        #logging.info((torch.autograd.grad(grad_ele[0,:].pow(2).sum(), X_adv,retain_graph=True)[0]).mean())
        if torch.isnan((torch.autograd.grad(grad_ele.pow(2).sum(), X_adv,retain_graph=True)[0]).max()) == torch.tensor(True).to(grad_ele.device):
            continue
        else:
            grads += (torch.autograd.grad(grad_ele.pow(2).sum(), X_adv,retain_graph=True)[0])
            #for kk in range(grad_ele.shape[0]):
            #    grads += (torch.autograd.grad(grad_ele[kk*1:(kk+1)*1,:].pow(2).sum(), X_adv,retain_graph=True)[0])
            #    logging.info(grads.mean())
            test_grads += (grad_ele.pow(2).sum()).detach()
    X_grad = grads
    X_adv  = X_adv.detach()

    return X_grad,loss,test_grads


def evaluate(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=config.output_path)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None


    pipeline = StableDiffusionPipeline.from_pretrained(config.model_id, torch_dtype=torch.float16).to(device)
    if config.sample.algorithm == 'ddim':
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    elif config.sample.algorithm == 'kdiffusion':
        pipeline.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    elif config.sample.algorithm == 'unipc':
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    else:
        pass

    #autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder = pipeline.vae
    autoencoder = autoencoder.to(device)
    unet = pipeline.unet
    unet = unet.to(device)
    #unet = accelerator.prepare(unet)

    mini_batch_size = config.train.batch_size // accelerator.num_processes
    dataset = IMAGE(config.dataset.image_path,config.dataset.height)
    train_dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True,
                                      num_workers=8, pin_memory=True, persistent_workers=True)
    epoch = len(dataset) // accelerator.num_processes
    logging.info(f'the number of epoch is {epoch}')
    
    unet,train_dataset_loader = accelerator.prepare(unet,train_dataset_loader)

    def get_data_generator():
        while True:
            for x,y,z in iter(train_dataset_loader):
                yield x,y,z

    data_generator = get_data_generator()

    #unet,autoencoder = accelerator.prepare(unet,autoencoder)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def decode_large_batch(_batch):
        decode_mini_batch_size = 50  # use a small batch size since the decoder is large
        xs = []
        pt = 0
        for _decode_mini_batch_size in utils.amortize(_batch.size(0), decode_mini_batch_size):
            x = decode(_batch[pt: pt + _decode_mini_batch_size])
            pt += _decode_mini_batch_size
            xs.append(x)
        xs = torch.concat(xs, dim=0)
        assert xs.size(0) == _batch.size(0)
        return xs

    logging.info(f'the config of sample is {config.sample}')
    logging.info(f'sample: n_samples={config.sample.n_samples}, mixed_precision={config.mixed_precision}')
    logging.info(f'the config of attacking is {config.attack}')

    def sample_z(init_image,cur_mask,cur_masked_image,_sample_steps, **kwargs):
        #_z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        
        ######## obtain the text embedding ########
        text_inputs = pipeline.tokenizer(
            config.sample.prompt,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_embeddings = pipeline.text_encoder(text_input_ids.to(pipeline.device))[0]

        uncond_tokens = [""]
        max_length = text_input_ids.shape[-1]
        uncond_input = pipeline.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
        seq_len = uncond_embeddings.shape[1]
        #text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        text_embeddings = uncond_embeddings
        text_embeddings = text_embeddings.detach()
        ######## obtain the text embedding ########

        mask = torch.nn.functional.interpolate(cur_mask, size=(config.dataset.height // 8, config.dataset.width // 8))
        mask = torch.cat([mask] * 2)
        X_adv   = init_image.clone().to(pipeline.device)
        z_shape = config.dataset.z_shape
        for i0 in tqdm(range(config.attack.iters)):
            all_grads = []
            losses = []
            all_test_grads = []
            for ii in range(config.attack.grad_reps):
                x_grad,loss,test_grads = poison_loss(unet,autoencoder,pipeline,
                                        X_adv,cur_mask,mask,text_embeddings,z_shape)
                all_grads.append(x_grad.cpu())
                all_test_grads.append(test_grads.cpu())
                losses.append(loss.cpu().detach().numpy())
                x_grad,loss,test_grads = 0,0,0
                X_adv = X_adv.detach()
                #logging.info(f'MeanGrad is {x_grad.cpu().mean().detach().numpy()}')

            grad = torch.stack(all_grads).mean(0).to(pipeline.device)
            if config.attack.algorithm == 'max':
                X_adv = X_adv + grad.detach().sign() * config.attack.step_size
            elif config.attack.algorithm == 'min':
                X_adv = X_adv - grad.detach().sign() * config.attack.step_size
            else:
                X_adv = X_adv + grad.detach().sign() * config.attack.step_size
            X_adv = torch.minimum(torch.maximum(X_adv, cur_masked_image - config.attack.eps), cur_masked_image + config.attack.eps) * (1 - cur_mask) + cur_mask * init_image.clone().to(pipeline.device)
            X_adv.data = torch.clamp(X_adv, min=config.attack.clamp_min, max=config.attack.clamp_max)
            X_adv = X_adv.detach()

            logging.info(f'{accelerator.is_main_process}--ites: {i0} ************************')
            logging.info(f'the grad of theta is {torch.stack(all_test_grads).mean(0).to(pipeline.device)}')
            logging.info(f'X_adv max: {(X_adv).max()} and X_adv min: {(X_adv).min()}')
            logging.info(f'AVG Loss: {np.mean(losses)}')
            nan_mask = torch.isnan(grad)
            nan_count = torch.sum(nan_mask)  
            total_size = grad.numel()  
            nan_ratio = nan_count / total_size
            logging.info(f'MaxGrad is {grad.cpu().max().detach().numpy()} and the nan counter {nan_ratio}')

            adv_Xf = (X_adv / 2 + 0.5).clamp(0, 1)
            to_pil_image = T.ToPILImage()
            image = to_pil_image(adv_Xf.squeeze())  # Squeeze the batch dimension
            image.save('./exp/poison_anna/test.png')
        return adv_Xf

    def sample_fn(init_image,cur_mask,cur_masked_image,):
        labels = None
        kwargs = dict()
        _z = sample_z(init_image,cur_mask,cur_masked_image,_sample_steps=config.sample.sample_steps, **kwargs)
        return _z

    if accelerator.is_main_process:
        os.makedirs(config.dataset.output_path, exist_ok=True)

    idx = 0
    to_pil_image = T.ToPILImage()
    for epoch_i in range(1):
        for kkk in range(1):
            init_image,cur_mask,cur_masked_image = next(data_generator)
        init_image,cur_mask,cur_masked_image = next(data_generator)
        cur_mask = cur_mask.to(pipeline.dtype).to(device)
        cur_masked_image = cur_masked_image.to(pipeline.dtype).to(device)
        init_image = init_image.to(pipeline.dtype).to(device)
        #init_image = prepare_image(init_image).to(device)
        logging.info(f'the shape and type of image is {init_image.shape} and {init_image.dtype}')
        image = sample_fn(init_image,cur_mask,cur_masked_image)
        images = accelerator.gather(image.contiguous())
        logging.info(f'the shape and type of image is {images.shape} and {images.dtype}')
        if accelerator.is_main_process:
            for sample in images:
                sample = to_pil_image(sample.squeeze())
                sample.save(os.path.join(config.dataset.output_path, f"{idx}.png"))
                idx += 1
        #image_out_dir = config.dataset.output_path + str(epoch_i) + '.png'
        #image.save(image_out_dir)
        #logging.info(f'Samples are saved in {image_out_dir}')

from absl import flags
from absl import app
from ml_collections import config_flags
import os


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "The nnet to evaluate.")
flags.DEFINE_string("output_path", None, "The path to output log.")


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    evaluate(config)


if __name__ == "__main__":
    app.run(main)

###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 