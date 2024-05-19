
# Appendix B in IQL paper
config = {
    "actor_lr": 3e-4,
    "value_lr": 3e-4,
    "critic_lr": 3e-4,
    "hidden_dims": (256, 256),
    "discount": 0.99,
    "expectile": 0.7,
    "beta": 3.0,
    "dropout_rate": 0.,
    "layer_norm": True,
    "tau": 0.005,  # exponential moving average

}
