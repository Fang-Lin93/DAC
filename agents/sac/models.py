import jax
import flax
import optax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from tensorflow_probability.substrates import jax as tfp
from flax import struct
from abc import ABC
import functools
from networks.mlp import MLP
from networks.initialization import orthogonal_init
from typing import Tuple, Sequence, Optional, Any, Callable

Params = flax.core.FrozenDict[str, Any]
PRNGKey = jax.random.PRNGKey
tfd = tfp.distributions


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)  # get log of temperature


class Actor(nn.Module):
    """
    As the action space is [0, 1], I use Normal-sigmoid policy
    """
    action_dim: int = 2
    hidden_dims: Sequence[int] = (32, 32)
    # should specify the type, otherwise it would be treated as the bound method nn.relu(self,...)
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 temperature: float = 1.0) -> tfd.Distribution:

        x = MLP(hidden_dims=self.hidden_dims,
                activations=self.activations,
                activate_final=True)(x)
        x = nn.Dense(self.action_dim, kernel_init=orthogonal_init())(x)

        log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

        return tfd.MultivariateNormalDiag(loc=nn.sigmoid(x),
                                          scale_diag=jnp.exp(log_stds) * temperature)


class Critic(nn.Module):
    hidden_dims: Sequence[int] = (32, 32)
    dropout_rate: Optional[float] = 0.1
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], -1)
        x = MLP(hidden_dims=self.hidden_dims,
                activations=self.activations,
                activate_final=True,
                dropout_rate=self.dropout_rate)(x)
        x = nn.Dense(1, kernel_init=orthogonal_init())(x)
        return jnp.squeeze(x, -1)


# class DoubleCritic(nn.Module):
#     # return a batch of q functions
#     hidden_dims: Sequence[int]
#     activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
#     num_qs: int = 2
#
#     @nn.compact
#     def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray):
#
#         VmapCritic = nn.vmap(Critic,
#                              variable_axes={'params': 0},
#                              split_rngs={'params': True},
#                              in_axes=None,
#                              out_axes=0,
#                              axis_size=self.num_qs)
#         qs = VmapCritic(self.hidden_dims,
#                         activations=self.activations)(observations, actions)
#         return qs


if __name__ == '__main__':
    from networks.critics import EnsembleQ
    actor = Actor(hidden_dims=(12, 16))
    actor.init(jax.random.PRNGKey(0), jnp.empty(3))
    multiQ = EnsembleQ(hidden_dims=(12, 16))
    v = multiQ.init(jax.random.PRNGKey(0), jnp.empty(3), jnp.empty(2))
    multiQ.apply(v, jnp.ones((10, 3)), jnp.ones((10, 2)))
