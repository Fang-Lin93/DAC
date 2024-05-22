"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from networks.policies import NormalTanhPolicy, sample_actions
from networks.types import Batch, InfoDict
from networks.model import Model
from networks.critics import EnsembleQ, ValueNet
from agents.base import Agent

from .updates import _update_jit_sql, _update_jit_eql


class IVRLearner(Agent):

    name = 'ivr'
    model_names = ["actor", "critic", "value", "target_critic"]

    def __init__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 seed: int,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.1,
                 dropout_rate: Optional[float] = None,
                 value_dropout_rate: Optional[float] = None,
                 layer_norm: bool = False,
                 lr_decay_steps: Optional[int] = None,
                 max_clip: Optional[int] = None,
                 # mix_dataset: Optional[str] = None,
                 alg: Optional[str] = None,
                 opt_decay_schedule: str = "cosine",
                 **kwargs):

        # self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.alpha = alpha
        self.max_clip = max_clip
        self.alg = alg

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = NormalTanhPolicy(hidden_dims,
                                     action_dim,
                                     log_std_scale=1e-3,
                                     log_std_min=-5.0,
                                     dropout_rate=dropout_rate,
                                     state_dependent_std=False,
                                     tanh_squash_distribution=False)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, lr_decay_steps)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             optimizer=optimiser)

        critic_def = EnsembleQ(hidden_dims, num_heads=2)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              optimizer=optax.adam(learning_rate=critic_lr))

        value_def = ValueNet(hidden_dims, layer_norm=layer_norm, dropout_rate=value_dropout_rate)
        value = Model.create(value_def,
                             inputs=[value_key, observations],
                             optimizer=optax.adam(learning_rate=value_lr))

        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = sample_actions(self.rng, self.actor.network,
                                      self.actor.params, observations,
                                      temperature, False)
        self.rng = rng

        actions = np.asarray(actions)
        return np.asarray(actions)

    def update(self, batch: Batch) -> InfoDict:
        # type <class 'str'> is not a valid JAX type.
        if self.alg == 'sql':
            new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit_sql(
                self.rng, self.actor, self.critic, self.value, self.target_critic,
                batch, self.discount, self.tau, self.alpha)
        elif self.alg == 'eql':
            new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit_eql(
                self.rng, self.actor, self.critic, self.value, self.target_critic,
                batch, self.discount, self.tau, self.alpha)
        else:
            raise NotImplementedError('please choose sql or eql')

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        return info
