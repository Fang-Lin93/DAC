
config = {
    "actor_lr": 3e-4,
    "value_lr": 3e-4,
    "critic_lr": 3e-4,
    "hidden_dims": (256, 256),
    "discount": 0.99,
    "dropout_rate": 0.,
    "layer_norm": True,
    "tau": 0.005,  # exponential moving average
    'alg': 'sql',  # 'sql' or 'eql'
    'alpha': 1.,

}

