import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 400123
    config.pretrained = d(
        model = "/2d-cfs-nj/aigc/model/stable-diffusion-xl-base-1.0"
    )
    config.revision = None
    config.run_name = "sdxl_lora"
    config.output_dir = "./temp/sdxl-lora3"
    config.logdir   = config.output_dir
    config.use_lora = True
    config.img_output_dir = '/2d-cfs-nj/alllanguo/code/test/exp_finetune2/datat-clean1024-n000050-lr3e_4-v2_1-6_255_sdxl'


    config.db_info = d(
        instance_data_dir = "/2d-cfs-nj/alllanguo/code/TEST/Attack/image/face_103",
        class_data_dir  =  "./class_img_sdxl",
        instance_prompt = "photo of a ktn person",
        class_prompt = "photo of a person",
        validation_prompt = "a ktn person in plane",
        num_validation_images = 1,
        validation_epochs = 100,
        with_prior_preservation = True,
        prior_loss_weight = 1.0,
        num_class_images  = 100,
        mixed_precision = 'fp16',
        use_xformers = True,
    )

    config.train = d(
        resolution = 1024,
        crops_coords_top_left_h = 0,
        crops_coords_top_left_w = 0,
        center_crop = True,
        train_text_encoder=False,
        train_batch_size  = 1,
        sample_batch_size = 1,
        max_train_steps = 3000,
        num_train_epochs= None,
        save_steps = 50,
        gradient_accumulation_steps = 1,
        gradient_checkpointing = True,
        lora_rank = 32,
        learning_rate = 2e-5,
        learning_rate_text = 1e-6,
        scale_lr = False,
        lr_scheduler = 'constant',
        lr_warmup_steps = 100,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_weight_decay = 1e-2,
        adam_epsilon  = 1e-8,
        max_grad_norm = 1,
        num_checkpoint_limit = 10,
        dataloader_num_workers = 4,
    )


    config.sample = d(
        num_steps=50,
        algorithm='dpm_solver',
        cfg=True,
        scale=5,
    )

    return config