export DATAR="CLEAN_IMAGE_FOLDER"
export DATA1="GENERATED_IMAGE_FOLDER"
python ism_fdfr.py --data_dir=$DATA1  --emb_dirs=$DATAR
python brisques.py --prompt_path=$DATA1
python -m pytorch_fid $DATAR $DATA1 
