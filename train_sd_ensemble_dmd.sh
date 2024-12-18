
accelerate launch train_sd_ensemble_dmd.py \
    --vad_output_dir "./train_cache/image/sd-vgg-ensemble_dmddit_12-255_sds" \
    --output_dir "./train_cache/pth2/sd-vgg-ensemble_dmddit_12-255_sds" \
    --data_json_file "eps-12_255-mom_anti-a9f0/VGGFace-all.json" \
    --pair_path "eps-12_255-mom_anti-a9f0/output_pairs.json" \
    --tensorboard_output_dir "logs/sd-vgg-ensemble_dmddit_12-255_10l1_all" \
    --resolution 512 > ./logs/sd-vgg-dmd_sds-12-255_sds.log 2>&1 &