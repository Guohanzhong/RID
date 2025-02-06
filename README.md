<p align="center">

  <h1 align="center">Real-time Identity Defenses against Malicious Personalization of Diffusion Models</h1>
</p>
  <p align="center">
    <a href="http://arxiv.org/abs/2412.09844"><img alt='page' src="https://img.shields.io/badge/Arxix-2412.09844-red"></a>
  </p>

# :label: News!

* Trained real-time identity defenses models are available ([https://drive.google.com/file/d/1qP1kcjz6fSzevWSEWFucJS7gDWk88-35/view?usp=sharing](https://drive.google.com/drive/folders/1EU49JpKiOy_IB4U0KdBuU-7k58WCi0JP?usp=share_link))here</a> to reproduce results in this paper.

# System Requirements

## Hardware Requirements

The `RID` library requires a GPU with at least 10 GB of VRAM for inference. If using CPU, one core is sufficient, and the RAM requirement is 16+ GB. For training, a GPU with 40 GB of VRAM or higher is necessary, along with at least 4 CPU cores (3.3 GHz or higher), and 16 GB of RAM. 

## Software Requirements

### OS Requirements

The development version of the package has been tested on *Linux* operating systems. The package is compatible with the following systems:

- **Linux**: Ubuntu 16.04
- **Mac OSX**: Supported
- **Windows**: Supported

Before setting up the package, users should have Python version 3.8 or higher installed, along with the necessary dependencies specified in the `requirements.txt` file. No need for any non-standard hardware.


# Installation Guide

### Install from Github
```
git clone https://github.com/Guohanzhong/RID
cd RID

# create an environment with python >= 3.8
conda create -n RID python=3.8
conda activate RID
pip install -r requirements.txt
```

# Demo

## RID inference to protect your images !
To carry out the defense on your own image, to run the following commands and changes the model path and images-folder path.
```sh
python infer.py -m 'model_path' -f 'folder_path' 
```
This will process a whole folder in 'folder_path' and save all the protected images in the '/output_folder/', the processing speed is 8 images per second when using A100.

## Personalization methods 
Based on the codebase of diffusers
```sh
cd tuning-based-personalization
accelerate launch --main_process_port $(expr $RANDOM % 10000 + 10000) train_sd_lora_dreambooth_token.py  --config=config/sd_lora.py  
```

# Training scripts

# Pseudocode
```pseudocode
Algorithm: Image Protection with RID
Input: 
    - images: Batch of normalized images ∈ [-1, 1] (shape [B, C, H, W])
    - RID_net: Pretrained perturbation generator
Output: 
    - protected_images: Protected images ∈ [-1, 1]

Process:
1. Generate perturbations:
    Δ = RID(images)  ▹ Matching shape [B, C, H, W]
    
2. Apply perturbations:
    protected = images + Δ
    
3. Clip to valid range:
    protected = clamp(protected, min=-1.0, max=1.0)

return protected
```
# :hearts: Acknowledgement

This project is heavily based on the [Diffusers](https://github.com/huggingface/diffusers) library, [DiT](https://github.com/facebookresearch/DiT) libary, [Anti-Dreambooth](https://github.com/VinAIResearch/Anti-DreamBooth) library.
Thanks for their great work!


# License
This project is covered under the **MIT License**.
