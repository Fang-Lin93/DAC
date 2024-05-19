

config = {
    # "actor_lr": 3e-4,
    # "critic_lr": 3e-4,
    "hidden_dims": (256, 256, 256),
    "dropout_rate": 0.,
    "layer_norm": False,
    "clip_grad_norm": 1.,
    "sampler": "ddpm",  # ddpm or ddim or ddpm_qg
    "ema_tau": 0.005,  # ema for critic_target learning and actor_target learning
    "update_ema_every": 5,
    "step_start_ema": 1000,
    # "eta": 10,  # BC + eta * guidance loss
    # "eta_lr": 0,  # if>0: using dual gradient ascent to update eta
    # "bc_threshold": 0.1,  # small -> BC, larger -> greedy
    # "use_soft_q": True,
    # "ra_scale": 0,  # if = 0: no ra weights;
    "use_guidance_loss": True,
    # "T": 5,
    "ddim_step": 5,
    "clip_sampler": True,  # set the same as DiffusionQL: clip the x0_hat in "ddpm - equation (7)"
    "beta_schedule": "vp",
    "action_prior": "normal",
    # "num_q_samples": 10,
    # "num_action_samples": 10,
    "act_with_q_guid": False,
    # "guidance_scale": 1,
    "action_argmax": True,  # take argmax Q (otherwise softmaxQ) for test actions
    # "use_cql": True,
    # "num_qs": 5,
    # "maxQ": False,
}