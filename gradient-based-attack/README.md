conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install accelerate==0.20.3 safetensors absl-py ml_collections einops ftfy==6.1.1 transformers==4.28.0

# xformers is optional, but it would greatly speed up the attention computation.
conda install xformers -c xformers