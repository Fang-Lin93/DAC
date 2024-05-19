

config = {
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "hidden_dims": (256, 256, 256),
    "dropout_rate": 0.,
    "layer_norm": False,
    "clip_grad_norm": 5.,
    "sampler": "ddpm",  # ddpm or ddim
    "tau": 0.001,  # ema for critic learning
    "eta": 1.,
    "T": 50,
    "ddim_step": 5,
    "clip_sampler": True,
    "beta_schedule": "vp",
    "action_prior": "normal",
    "num_q_samples": 10,
    "num_action_samples": 50,
    "maxQ": False,
    # "bc_path": None,
}