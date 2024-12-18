

## FDFR and ISM
Set up env for FDFR and ISM. Firstly, we need to `cd evalutions` and install the following library:

```
cd deepface
pip install -e .

cd retinaface
pip install -e .
```

Note: You should follow the tensorflow instruction from `https://www.tensorflow.org/install/pip` to accelerate evaluation process (without tensorflow, the eval code will run on cpu which is slow)

To compute FDFR and ISM. If you get block by proxy, manually download weight of DeepFace from https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5 and download weight from https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5, place it in folder "{home}/.deepface/weights/".
We run the following command to compute FDFR and ISM soore: `python evaluations/ism_fdfr.py --data_dir <path_to__perturb_image_dir> --emb_dirs <paths_to_id_emb_file>`



## Brisque

1. `pip install brisque`
2. Run `python brisques.py --prompt_path <path_to_perturb_images_of_prompts>`


## FID

1. `pip install pytorch_fid`
2. Run `python -m pytorch_fid <path_to__perturb_image_dir> <paths_to_id_emb_file>`

