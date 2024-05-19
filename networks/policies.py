
import functools
import jax
import flax.linen as nn
import numpy as np
import jax.numpy as jnp
from networks.types import Params, PRNGKey
from typing import Optional, Sequence, Tuple
from tensorflow_probability.substrates import jax as tfp
from networks.mlp import MLP
from networks.initialization import orthogonal_init
from networks.distributions.tanh_transformed import TanhTransformedDistribution


tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class MSEPolicy(nn.Module):
    # flax module
    # 1. defines parameters at the beginning of the name here
    # 2. it's a skeleton only, it should be used by net.apply({'params': params}, x, ...)
    # 3.
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    layer_norm: bool = False
    deterministic: bool = True

    @nn.compact  # use with .apply({'params': params}, *args, **kwargs)
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> jnp.ndarray:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      layer_norm=self.layer_norm,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        actions = nn.Dense(self.action_dim,
                           kernel_init=orthogonal_init())(outputs)
        return nn.tanh(actions)


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = False
    dropout_rate: Optional[float] = None
    layer_norm: bool = False
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True
    deterministic: bool = False

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      layer_norm=self.layer_norm,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)
        means = nn.Dense(self.action_dim, kernel_init=orthogonal_init())(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=orthogonal_init(
                                    self.log_std_scale))(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim, ))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)
        if self.tanh_squash_distribution:
            # return tfd.TransformedDistribution(distribution=base_dist,
            #                                    bijector=tfb.Tanh())
            return TanhTransformedDistribution(base_dist)
        else:
            return base_dist


# be careful with static_argnames...
@functools.partial(jax.jit, static_argnames=('network', 'deterministic'))
def _sample_actions(rng: PRNGKey,
                    network: nn.Module,
                    params: Params,
                    observations: np.ndarray,
                    temperature: float = 1.0,
                    deterministic: bool = False) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = network.apply(params, observations, temperature)
    rng, key = jax.random.split(rng)

    if deterministic:
        return rng, dist
    return rng, dist.sample(seed=key)


def sample_actions(rng: PRNGKey,
                   network: nn.Module,
                   params: Params,
                   observations: np.ndarray,
                   temperature: float = 1.0,
                   deterministic: bool = True) -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(rng, network, params, observations, temperature, deterministic)
