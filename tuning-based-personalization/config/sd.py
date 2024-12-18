import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 3023
    config.pretrained = d(
        model = "stable-diffusion-2-1-base"
    )
    config.revision = None
    config.run_name = "sd"
    config.output_dir = "train_cache/exp/sd-15-gam-8_255-db"
    config.logdir   = config.output_dir
    config.use_lora = True
    config.img_output_dir = 'train_cache/image/gam-v2_1-8_255-15-db'


    config.db_info = d(
        instance_data_dir = "attack/eps8-255-gam/15",
        class_data_dir  =  "./image/class_people_sd_new",
        # <new1>
        instance_prompt = "photo of sks person",
        class_prompt = "photo of person",
        #validation_prompt = "a dslr portrait of sks person",
        #validation_prompt = "a dslr portrait of sks person",
        validation_prompt = "sks person with a dog",
        #validation_prompt = "photo of a <new1> person in the beach",
        #validation_prompt = "a close-up photo of sks person, high details",
        num_validation_images = 1,
        validation_epochs = 1000,
        with_prior_preservation = True,
        prior_loss_weight = 1.0,
        num_class_images  = 100,
        mixed_precision = 'fp16',
        use_xformers = False,
        #modifier_token = "<new1>",
        modifier_token = None,
        initializer_token = "ktn+pll+ucd",
    )

    config.train = d(
        resolution = 512,
        crops_coords_top_left_h = 0,
        crops_coords_top_left_w = 0,
        center_crop = True,
        train_text_encoder=True,
        train_batch_size  = 3,
        sample_batch_size = 1,
        max_train_steps = 6001,
        num_train_epochs= None,
        save_steps = 1000,
        gradient_accumulation_steps = 1,
        gradient_checkpointing = False,
        learning_rate = 8e-7,
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