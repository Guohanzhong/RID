import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 4033223
    config.pretrained = d(
        model = "stable-diffusion-2-1-base"
    )
    config.revision = None
    config.run_name = "sd_lora"
    config.output_dir = "train_cache/exp/gam"
    config.logdir   = config.output_dir
    config.use_lora = True
    config.img_output_dir = 'train_cache/image/gam'


    config.db_info = d(
        instance_data_dir =  "attack/eps12-255-gam/04",
        class_data_dir  =  "./image/class_people_sd_new",
        # <new1>
        instance_prompt = "photo of a <new1> person",
        class_prompt = "photo of a person",
        validation_prompt = "<new1> person wears a glass",
        #validation_prompt = "a dslr portrait of <new1> person",
        #validation_prompt = "a close-up photo of <new1> person, high details",
        num_validation_images = 1,
        validation_epochs = 10,
        with_prior_preservation = True,
        prior_loss_weight = 1.0,
        num_class_images  = 100,
        mixed_precision = 'fp16',
        use_xformers = False,
        modifier_token = "<new1>",
        initializer_token = "ktn+pll+ucd",
    )

    config.train = d(
        resolution = 512,
        crops_coords_top_left_h = 0,
        crops_coords_top_left_w = 0,
        center_crop = True,
        train_text_encoder=False,
        train_batch_size  = 1,
        sample_batch_size = 1,
        max_train_steps = 250,
        num_train_epochs= None,
        save_steps = 10,
        gradient_accumulation_steps = 1,
        gradient_checkpointing = False,
        lora_rank = 32,
        #learning_rate = 5e-5,
        learning_rate = 3e-4,
        scale_lr = False,
        lr_scheduler = 'constant',
        lr_warmup_steps = 0,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_weight_decay = 1e-2,
        adam_epsilon  = 1e-8,
        max_grad_norm = 1,
        num_checkpoint_limit = 300,
        num_train_timesteps  = 1000,
        dataloader_num_workers = 4,
    )


    config.sample = d(
        num_steps=50,
        algorithm='dpm_solver',
        cfg=True,
        scale=5,
    )

    return config