import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.output_path = 'demo.log'
    config.model_id = "stabilityai/stable-diffusion-2-1-base"

    config.sample = d(
        algorithm = 'ddim',
        cfg=True,
        mini_batch_size=1,
        n_samples=1,
        prompt='',
        sample_steps=12,
        scale=7,
        strength=0.8,
    )

    config.train = d(
        batch_size = 1,
        iterations = 1000,
    )

    config.dataset = d(
        image_path = '',
        height=512,
        width=512,
        output_path = '',
        z_shape = (1, 4, 64, 64)
    )

    config.attack = d(
        iters=100,
        grad_reps=10,
        step_size=1/255,
        eps=16/255,
        clamp_min=-1,
        clamp_max=1,
        algorithm='max',
    )


    return config