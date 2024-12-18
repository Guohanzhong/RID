accelerate launch train_sd_ensemble_reg.py \
    --vad_output_dir "./train_cache/image/sd-vgg-civitai-ensemble_dmddit_6-255_l2_reg" \
    --output_dir "./train_cache/pth/sd-vgg-civitai-ensemble_dmddit_6-255_l2_onlyreg" \
    --data_json_file "eps-6_255-/VGGFace-all.json" \
    --pair_path "eps-6_255-/output_pairs.json" \
    --tensorboard_output_dir "logs/0227_dmd-dit_reg" \
    --resolution 512 > ./logs/sd-vgg-civitai-dmd_reg.log 2>&1 &