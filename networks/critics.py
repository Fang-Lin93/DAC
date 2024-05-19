from networks.types import Callable, Sequence, Optional, PRNGKey

import jax.numpy as jnp
from flax import linen as nn

from jax import random
from networks.mlp import MLP
from networks.initialization import orthogonal_init
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class ValueNet(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        state_value = MLP((*self.hidden_dims, 1),
                          layer_norm=self.layer_norm,
                          dropout_rate=self.dropout_rate)(observations)
        return jnp.squeeze(state_value, -1)


class QNet(nn.Module):
    # input concatenate(s, a) -> scalar
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     layer_norm=self.layer_norm,
                     dropout_rate=self.dropout_rate,
                     activations=self.activations)(inputs)

        return critic.squeeze()  # (B, 1) -> (B,)
        # return jnp.squeeze(critic, -1)  # (B, 1) -> (B,)


class NormalQNet(nn.Module):
    # input concatenate(s, a) -> scalar
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    layer_norm: bool = False
    dropout_rate: Optional[float] = None
    state_dependent_std: bool = False

    @nn.compact
    def __call__(self,
                 rng: PRNGKey,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 temperature: float = 1.) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        outputs = MLP(self.hidden_dims,
                      layer_norm=self.layer_norm,
                      dropout_rate=self.dropout_rate,
                      activations=self.activations)(inputs)

        means = nn.Dense(1, kernel_init=orthogonal_init())(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(1,
                                kernel_init=orthogonal_init(
                                    self.log_std_scale))(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (1,))  # set to zero

        base_dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        return base_dist


class EnsembleQ(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    layer_norm: bool = False
    dropout_rate: Optional[float] = None
    num_qs: int = 2

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray):
        VmapQ = nn.vmap(QNet,
                        variable_axes={'params': 0},
                        split_rngs={'params': True},
                        in_axes=None,
                        out_axes=0,
                        axis_size=self.num_qs)
        qs = VmapQ(hidden_dims=self.hidden_dims,
                   activations=self.activations,
                   dropout_rate=self.dropout_rate,
                   layer_norm=self.layer_norm,
                   )(observations, actions)

        return qs


if __name__ == '__main__':
    import jax
    import optax
    import re
    from functools import partial
    from networks.model import Model, get_weight_decay_mask

    qs_fn = EnsembleQ((5, 5))
    params = qs_fn.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)), jnp.zeros((1, 2)))

    key_ = jax.random.PRNGKey(1)

    qs_ = qs_fn.apply(params, jax.random.normal(key_, (256, 3)), jax.random.normal(key_, (256, 2)))

    critic = Model.create(EnsembleQ((5, 5), layer_norm=False),
                          inputs=[key_, jnp.zeros((1, 3)), jnp.zeros((1, 2))],  # time
                          optimizer=optax.adamw(learning_rate=3e-4, weight_decay=1e-4,
                                                mask=get_weight_decay_mask),
                          clip_grad_norm=True)
