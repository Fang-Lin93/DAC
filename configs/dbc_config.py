

config = {
    "actor_lr": 3e-4,
    "hidden_dims": (256, 256, 256),
    "dropout_rate": 0.,
    "layer_norm": False,
    "sampler": "ddpm",  # or "ddim"
    "T": 5,
    "clip_sampler": False,  # try this
    "ddim_step": 5,
    "beta_schedule": "vp",
}
