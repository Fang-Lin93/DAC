from typing import Tuple, Sequence, Union, Optional
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from networks.policies import NormalTanhPolicy, sample_actions
from datasets import Batch
from networks.model import Model
from networks.types import InfoDict, Params, PRNGKey
from agents.base import Agent

"""Implementations of algorithms for continuous control."""


def get_weight_decay_mask(params):
    flattened_params = flax.traverse_util.flatten_dict(
        flax.core.frozen_dict.unfreeze(params))

    def decay(k):
        if any([(key == 'bias' or 'Input' in key or 'Output' in key)
                for key in k]):
            return False
        else:
            return True

    return flax.core.frozen_dict.freeze(
        flax.traverse_util.unflatten_dict(
            {k: decay(k)
             for k, v in flattened_params.items()}))


# _jit_mse_update = jax.jit(mse_update)

# MSE update function
@jax.jit
def _mse_update(actor: Model, batch: Batch, rng: PRNGKey) -> Tuple[Model, InfoDict]:
    rng, key = jax.random.split(rng)

    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # it creates a function of params, so it has to directly use net.apply() function rather than
        # the __call__() function of the Model object "actor"
        # here training is True for this forward pass is used for gradient descent
        actions = actor.apply(actor_params,
                              batch.observations,
                              training=True,
                              rngs={'dropout': key})
        #
        actor_loss = ((actions - batch.actions) ** 2).mean()
        return actor_loss, {'actor_loss': actor_loss}

    return rng, *actor.apply_gradient(loss_fn)  # *(Model, info) -> Model, info


@jax.jit
def _mle_update(actor: Model, batch: Batch, rng: PRNGKey, entropy_bonus=None) -> Tuple[Model, InfoDict]:
    rng, key1, key2 = jax.random.split(rng, 3)

    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply(
            actor_params,
            batch.observations,
            training=True,
            rngs={"dropout": key1},
        )

        # here is learning using mle: for normal policy it is equivalent to MSE
        nll = -dist.log_prob(batch.actions).mean()
        actor_loss = nll

        action = dist.sample(seed=key2)
        eps = 1e-5
        action = jax.lax.stop_gradient(jnp.clip(action, -1 + eps, 1 - eps))
        log_prob = dist.log_prob(action)

        if entropy_bonus is not None:
            entropy_grad = (-log_prob * jax.lax.stop_gradient(log_prob)).mean()

            actor_loss -= entropy_bonus * entropy_grad

        return actor_loss, {"nll": nll, "entropy": -log_prob.mean()}

    return rng, *actor.apply_gradient(loss_fn)  # *(Model, info) -> Model, info


class BCLearner(Agent):

    save_dir = 'ckpt/bc'
    model_names = ['actor']

    def __init__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 seed: int,
                 actor_lr: Union[float, optax.Schedule] = 1e-3,
                 hidden_dims: Sequence[int] = (256, 256),
                 layer_norm: bool = False,
                 dropout_rate: Optional[float] = None,
                 weight_decay: Optional[float] = None,
                 entropy_bonus: Optional[float] = None,
                 lr_decay_steps: int = 1000000, **kwargs):
        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng)

        action_dim = actions.shape[-1]

        network = NormalTanhPolicy(hidden_dims,
                                   action_dim,
                                   log_std_scale=1e-3,
                                   log_std_min=-5.0,
                                   dropout_rate=dropout_rate,
                                   state_dependent_std=False,
                                   tanh_squash_distribution=True,
                                   layer_norm=layer_norm)

        if weight_decay is None:
            optimizer = optax.adam(learning_rate=actor_lr)
        else:
            # -actor_lr is used for gradient descent: theta =theta - lr * grad
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, lr_decay_steps)
            optimizer = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))

        self.actor = Model.create(network,
                                  # RNG, observations for network initialization
                                  inputs=[actor_key, observations],
                                  optimizer=optimizer)
        self._obs_dummy = jnp.ones_like(observations)
        self.rng = rng
        self.entropy_bonus = entropy_bonus

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 0) -> jnp.ndarray:
        self.rng, actions = sample_actions(self.rng,
                                           self.actor.network,
                                           self.actor.params,
                                           observations,
                                           temperature,
                                           deterministic=False)
        return np.asarray(actions).clip(-1, 1)

    def update(self, batch: Batch) -> InfoDict:
        self.rng, self.actor, info = _mle_update(self.actor, batch, self.rng, self.entropy_bonus)
        # self.rng, self.actor, info = _mse_update(self.actor, batch, self.rng)
        return info

    def summary(self):
        print(self.actor.network.tabulate(self.rng, self._obs_dummy))
