import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 4033223
    config.pretrained = d(
        #model = "runwayml/stable-diffusion-v1-5"
        #model = "stabilityai/stable-diffusion-2-1-base"
        model =  "/2d-cfs-nj/aigc/model/stable-diffusion-2-1-base"
        #model = "/mnt_alipayshnas/wuyou.zc/models/diffusions/stable-diffusion-v1-4"
    )
    config.revision = None
    config.run_name = "sd_lora"
    config.output_dir = "./temp/test_sdstyle2"
    config.logdir   = config.output_dir
    config.use_lora = True
    #config.img_output_dir = '/2d-cfs-nj/alllanguo/code/test/exp_finetune/style-vingo-lr3e-4'
    config.img_output_dir = '/2d-cfs-nj/alllanguo/code/test/exp_finetune/Ensem_3c_Uni_5e-7_8t255_n100_t0t900_sn16_f0a10_datat-vingo_lr3e-4'


    config.db_info = d(
        #instance_data_dir = "/2d-cfs-nj/alllanguo/code/test/Pyguard/image/style_paint",
        instance_data_dir = "/2d-cfs-nj/alllanguo/code/test/Attack/EXP-new/Ensem_3_Uni_attack_5e-7_8t255_n100_t0t900_sn16_f0a10v_data-vinstyle/noise-ckpt/70",
        class_data_dir  =  "./image/class_style_paint",
        instance_prompt = "a <new1> style painting",
        class_prompt = "a realistic painting",
        validation_prompt = "a <new1> style painting, with a little boy",
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
        max_train_steps = 3000,
        num_train_epochs= None,
        save_steps = 10,
        gradient_accumulation_steps = 1,
        gradient_checkpointing = False,
        lora_rank = 32,
        learning_rate = 3e-4,
        scale_lr = False,
        lr_scheduler = 'constant',
        lr_warmup_steps = 0,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_weight_decay = 1e-2,
        adam_epsilon  = 1e-8,
        max_grad_norm = 1,
        num_checkpoint_limit = 30,
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