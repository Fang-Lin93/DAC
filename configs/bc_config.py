

config = {
    "actor_lr": 3e-4,
    "hidden_dims": (256, 256),
    "dropout_rate": 0.,
    "layer_norm": True,
}


# import ml_collections
#
#
# def get_config():
#     config = ml_collections.ConfigDict()
#
#     # config.distribution = 'det'
#     # det (deterministic) or mog (mixture of gaussians) or made_mog or made_d (discretized)
#
#     config.actor_lr = 1e-3
#     config.hidden_dims = (256, 256)
#
#     return config
