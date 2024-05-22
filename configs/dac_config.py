

config = {
    "hidden_dims": (256, 256, 256),
    "dropout_rate": 0.,
    "layer_norm": False,
    "clip_grad_norm": 1.,
    "sampler": "ddpm",  # ddpm or ddim or ddpm_qg
    "ema_tau": 0.005,  # ema for critic_target learning and actor_target learning
    "update_ema_every": 5,
    "step_start_ema": 1000,
    "use_guidance_loss": True,
    "ddim_step": 5,
    "clip_sampler": True,  # set the same as DiffusionQL: clip the x0_hat in "ddpm - equation (7)"
    "beta_schedule": "vp",
    "action_prior": "normal",
    "act_with_q_guid": False,
}