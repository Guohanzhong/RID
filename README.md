<p align="center">

  <h1 align="center">Real-time Identity Defenses against Malicious Personalization of Diffusion Models</h1>
</p>
  <p align="center">
    <a href="http://arxiv.org/abs/2412.09844"><img alt='page' src="https://img.shields.io/badge/Arxix-2412.09844-red"></a>
  </p>

# :label: News!

* Trained real-time identity defenses models are available ([https://drive.google.com/file/d/1qP1kcjz6fSzevWSEWFucJS7gDWk88-35/view?usp=sharing](https://drive.google.com/drive/folders/1EU49JpKiOy_IB4U0KdBuU-7k58WCi0JP?usp=share_link))here</a> to reproduce results in this paper.

## 🔎 Overview framework
![seesr](asset/frame.png)


# System Requirements

## Hardware Requirements

The `RID` library requires a GPU with at least 3 GB of VRAM for inference with fp32 and at least 2GB of VRAM for inference with fp16 with single batch. If using CPU, one core is sufficient, and the RAM requirement is 3GB. For training, a GPU with 40 GB of VRAM or higher is necessary, along with at least 4 CPU cores (3.3 GHz or higher), and 16 GB of RAM. 

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

# Inference Demo

## RID inference to protect your images !
To carry out the defense on your own image, to run the following commands and changes the model path and images-folder path.
```sh
python infer.py -m 'model_path' -f 'folder_path' 
```
or
```sh
bash infer_test.sh
```
This will process a whole folder in 'folder_path' and save all the protected images in the '/output_folder/', the processing speed is 8 images per second when using A100.
'model_path' is the checkpoint of RID network, which can download from ([Google Drive](https://drive.google.com/drive/folders/1EU49JpKiOy_IB4U0KdBuU-7k58WCi0JP?usp=share_link)).

While the inference of optional perturbation purification in the same while changing the model path and input folder. It is required to run the protection using the RID and using the output folder as the input folder in the second stage.

Meanwhile, to implement the RID and personalization on certrain ID, we provide a folder of one person in './asset/15_supp' extracted from the evaluation dataset.

## Evaluation the performance of protection
### Personalization methods 
In order to evaluation the protection performance, based on [Diffusers](https://github.com/huggingface/diffusers), run the personalization using the following commands
```sh
cd tuning-based-personalization
accelerate launch --main_process_port $(expr $RANDOM % 10000 + 10000) train_sd_lora_dreambooth_token.py  --config=config/sd_lora.py  
```
'config/sd_lora.py' contains the parameters needed for personalization training. ‘train_sd_lora_dreambooth_token.py' is used for training the 'LoRA+TI', costs about 10 minutes to train. While set 'config.use_lora = False' in 'config/sd_lora.py', the personalization method becomes 'TI', which costs 5 minutes to train.

The following command is use the 'DB' as the personalization method, which costs 30 minutes to train.
```sh
cd tuning-based-personalization
accelerate launch --main_process_port $(expr $RANDOM % 10000 + 10000) train_sd_dreambooth_token.py  --config=config/sd.py  
```

### More results about the robustness

In the Extended Data Figure~3, Extended Data Figure~4, Extended Data Figure~5, Extended Data Figure~6 in the Main text, we provide robustness of RID across a wide range of conditions (including a larger number of identities, diverse gender, and racial groups) and mixed clean/protected training data settings. 
To better help to reproduce the results in the paper, we provide the data of different IDs used in evaluation.

Due to some potential privacy issues, we only put up mixed results for the evaluation set used by [115 IDs](https://drive.google.com/file/d/14rsm9tZdkXhuMPqTyYaQm2LztL7AkTKQ/view?usp=share_link) and the mixed clean/protected training data settings.


### Results after post-processing on defended images.

In the real-world scenarios, adversaries may apply various post-processing techniques to protected images to weaken the defense before launching personalized image generation, which we already considered in the page 12 in the Main text, to better reproduce the results of Figure~6 in the Main text, we provide code repositories for different post-processing methods. For any post-processing approach, simply apply their processing to the defended images obtained from RID inference to generate post-processed defended images.

(1) JPEG compression (JPEG-C), a traditional approach that compresses high-frequency image information, potentially diminishing the effectiveness of added perturbations; 
```sh
from PIL import Image
img = Image.open(image_path)
img.save(output_path, "JPEG", quality=75)  
```
Or run the follwing commands where '-i' denotes the input folder after protection.
```sh
python post_j.py -i -o -q 75
```

(2) [DiffPure](https://github.com/NVlabs/DiffPure), a diffusion-based method that applies noise and then denoises defended images, leveraging the generative capacity of diffusion models to restore clean features. 
We implement the Diffpure based on the diffusers, running in
```sh
python post_diffpure.py \
    -m "path/to/your/stable-diffusion-model" \
    -i "path/to/defended_images_base_folder" \
    -o "path/to/output_purified_images_folder" \
    -p "01,02,03" \
    -s 0.1
```
where '-m' denotes '--model_path', the path to the Stable Diffusion model (required). '-i' denotes '--input_dir', the root directory containing subfolders of protected images (required). '-o' denotes '--output_dir', the root directory to save the processed images (required). '-p' denotes '--process_list', the omma-separated list of subfolder names to process. If omitted, all subfolders will be processed. '-s' denotes '--strength', denoising strength, controlling how much the image is altered (default: 0.1).


(3) [GridPure](https://github.com/ZhengyueZhao/GrIDPure), which processes 256×256 image patches independently using a pre-trained diffusion model~\cite{dhariwal2021diffusion} to locally denoise defended images like Diffpure;

In order to implement the GridPure, it is required to run
```sh
git clone https://github.com/ZhengyueZhao/GrIDPure.git
cd GrIDPure
python gridpure.py \
    --input_dir="" \
    --output_dir="" \
    --pure_model_dir="" \
    --pure_steps=10 \
    --pure_iter_num=20 \
    --gamma=0.1
```
which 'input_dir' denotes the defended image folder, and 'output_dir' is the output path for postprocessing under the GridPure. 'pure_model_dir' denotes the model path for '256x256_diffusion_uncond.pt'.
While the checkpoint can be downloaded via 
```sh
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
```
It is worth noting that there are some typos in [GridPure](https://github.com/ZhengyueZhao/GrIDPure), to run sucessfully, you should change the line-138 in 'gridpure.py' to
```sh
diffpure_model_dir = args.pure_model_dir
```

(4) [Noiseup](https://github.com/ethz-spylab/robust-style-mimicry), a more aggressive purification strategy involving upsampling via a large-scale super-resolution model (from Stable Diffusion) followed by downsampling.

In order to implement the Noiseup, it is required to run
```sh
git clone https://github.com/ethz-spylab/robust-style-mimicry.git
cd robust-style-mimicry
python noise.py --in_dir --out_dir  --gaussian_noise 0.05
python upscale.py --in_dir --out_dir  
```
It worth noting that to run sucessfully, you should replace the 'noise.py' 'upscale.py' with there two files in './robust-style-mimicry'


Many thanks for their work.

# Training scripts

## Prepare the dataset

In order to train RID, we need to first prepare the training dataset, which consists of the original dataset as well as protected image-image data pairs constructed using a gradient-based approach.

The pairs data is important for training the RID. To generate the pairs data, we use the [Anti-Dreambooth](https://github.com/VinAIResearch/Anti-DreamBooth) library to generate the corresponding perturbation for each image and store these pairs for the following training.

```sh
cd gradient-based-attack
accelerate launch --num_processes 1 aspl_ensemble.py --pretrained_model_name_or_path "" --instance_data_dir_for_train "" --instance_data_dir_for_adversarial ""  --pgd_eps "" --output_dir ""
```
where 'pretrained_model_name_or_path' denotes the pre-trained models use for generation pairs data, it should be aligned with the process of training the RID in the following.
'instance_data_dir_for_train' and 'instance_data_dir_for_adversarial' denote the same clean images folder, while the images are directly stored in this folder.
'output_dir' denotes the output dir for protected images.
'pgd_eps' denotes the perturbation scale, with higher scale, the protection is more visible. We recommand to set the perturbation scale as 12/255.

## Training the RID

After preparing the dataset, run the following commands to train the RID,
```sh
sh train_sd_ensemble_dmd.sh
```
or
```
accelerate launch train_sd_ensemble_dmd.py \
    --pretrained_model_name_or_path ""
    --vad_output_dir "./train_cache/image/sd-vgg-ensemble_dmddit_12-255_sds" \
    --output_dir "./train_cache/pth2/sd-vgg-ensemble_dmddit_12-255_sds" \
    --data_json_file "eps-12_255-mom_anti-a9f0/VGGFace-all.json" \
    --pair_path "eps-12_255-mom_anti-a9f0/output_pairs.json" \
    --tensorboard_output_dir "logs/sd-vgg-ensemble_dmddit_12-255_10l1_all" \
    --resolution 512 > ./logs/sd-vgg-dmd_sds-12-255_sds.log 2>&1 &
```

where 'pretrained_model_name_or_path' denotes the pre-trained diffusion models we use in training, to use the ensemble models to train, set 'pretrained_model_name_or_path' as 'model_1,model_2,model_3'.

'data_json_file' denotes the a JSON file that stores a list of dictionaries. Each dictionary in this list must have at least one key "image_file", which represents the file path of an image. In our paper, we use the VGGFace2 as the raw dataset.
For instance, 
```
[
    {"image_file": "1.png"},
    {"image_file": "2.png"},
]
```
The 'pair_path' should also be a JSON array where each element is a JSON object. Each object must contain two keys: "source_path" and "attacked_path". The value corresponding to "source_path" is the file path of the source image, and the value corresponding to "attacked_path" is the file path of the protected image which is generated using the Anti-DB. 
For instancce, 
```
[
    {"source_path": "source_1.png", "attacked_path": "attacked_1.png"},
    {"source_path": "source_2.png", "attacked_path": "attacked_2.png"},
]
```
The order and number of elements do not need to be the same for both jsons.

'output_dir' denotes the the output dir for the trained RID network. 'vad_output_dir' represents the output of RID during training.

We also provide the ability to optimize the RID using only regression, using the following command,
```sh
sh train_sd_ensemble_reg.sh
```
Training RID costs about 7 days with 8 A100-40G.

# Pseudocode
### Pseudocode of Inference
```pseudocode
Inference Algorithm: Image Protection with RID
Input: 
    - images: Batch of normalized images ∈ [-1, 1] (shape [B, C, H, W])
    - RID_net: Pretrained perturbation generator
Output: 
    - protected_images: Protected images ∈ [-1, 1]

Process:
1. Generate perturbations:
    # Real-time protection
    Δ = RID(images)  ▹ Matching shape [B, C, H, W]
    
2. Apply perturbations:
    protected = images + Δ
    
3. Clip to valid range:
    protected = clamp(protected, min=-1.0, max=1.0)

return protected
```
### Pseudocode of our RID library
```
└── infer.py                      ## inference code using trained RID
└── train_sd_ensemble_dmd.sh      ## training scripts using Adv-SDS
└── train_sd_ensemble_dmd.py      ## training code using Adv-SDS
└── train_sd_ensemble_reg.sh      ## training scripts using only regression
└── train_sd_ensemble_reg.py      ## training code using only regression
└── tuning-based-personalization/ ## the peronsonalization code
    └── train_sd_dreambooth_token.py        ## train DB
    └── train_sd_lora_dreambooth_token.py    ## train Lora+TI/TI
    └── config   ## config using in training personalization
        └── sd_lora.py    # config for Lora+TI/TI
        └── sd.py         # config for DB
    └── ...
└── gradient-based-attack/         ## the code for generation pairs data using gradient-based protection methods
    └── aspl_ensemble.py           ## gradient-based protection code
    └── ...
└── evaluation/                    ## Quantitative evaluation code
```


# User study

Given the limitations of quantitative metrics and the subjective nature of the task in our paper, we conducted a carefully designed user study to assess both the protection effectiveness [(User study 1)]() and visual imperceptibility [(User study 2)]() of RID in comparison with baseline approaches. 
The results of these two user studies are shown in the Figure~2e and Figure~2f in the Main text.

Further, due to the misleading quantitative results of the [Noiseup](https://github.com/ethz-spylab/robust-style-mimicry), we carried out another user study about the protection effectiveness between the undefended images and defended images with the Noiseup. [(User study 3)](), whose results are shown in Figure~S.2 in Supplementary
Note C in Supplementary Information.

# :hearts: Acknowledgement

This project is heavily based on the [Diffusers](https://github.com/huggingface/diffusers) library, [DiT](https://github.com/facebookresearch/DiT) libary, [Anti-Dreambooth](https://github.com/VinAIResearch/Anti-DreamBooth) library.
Thanks for their great work!


# License
This project is covered under the **MIT License**.
