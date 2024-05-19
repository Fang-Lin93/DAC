"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import os
import jax
import jax.numpy as jnp
import numpy as np
import optax

# from agents.agent import Agent
from .updates import _update_jit
from networks.policies import NormalTanhPolicy, sample_actions
from networks.model import Model
from networks.critics import ValueNet, EnsembleQ
from networks.types import InfoDict, Batch
from agents.base import Agent

# from critic import update_q, update_v


class IQLLearner(Agent):

    name = "iql"
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
                 tau: float = 0.005,  # lr for exponential moving avg. (soft updates)
                 expectile: float = 0.8,
                 beta: float = 1,  # inverse temperature for awr
                 temperature: float = 0.1,
                 dropout_rate: Optional[float] = None,
                 layer_norm: bool = False,
                 lr_decay_steps: Optional[int] = None,
                 opt_decay_schedule: str = "cosine",
                 **kwargs):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        self.expectile = expectile
        self.beta = beta
        self.tau = tau
        self.discount = discount
        self.temperature = temperature

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_net = NormalTanhPolicy(hidden_dims,
                                     action_dim,
                                     log_std_scale=1e-3,
                                     log_std_min=-5.0,
                                     dropout_rate=dropout_rate,
                                     state_dependent_std=False,
                                     tanh_squash_distribution=False)

        if opt_decay_schedule == "cosine" and lr_decay_steps is not None:
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, lr_decay_steps)
            act_opt = optax.chain(optax.scale_by_adam(),
                                  optax.scale_by_schedule(schedule_fn))
        else:
            act_opt = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_net,
                             inputs=[actor_key, observations],
                             optimizer=act_opt)

        DoubleQ_net = EnsembleQ(hidden_dims, num_heads=2)
        critic = Model.create(DoubleQ_net,
                              inputs=[critic_key, observations, actions],
                              optimizer=optax.adam(learning_rate=critic_lr))

        value_def = ValueNet(hidden_dims, layer_norm=layer_norm, dropout_rate=dropout_rate)
        value = Model.create(value_def,
                             inputs=[value_key, observations],
                             optimizer=optax.adam(learning_rate=value_lr))

        target_critic = Model.create(
            DoubleQ_net, inputs=[critic_key, observations, actions])

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1) -> jnp.ndarray:
        rng, actions = sample_actions(self.rng, self.actor.network,
                                      self.actor.params, observations,
                                      temperature, False)
        self.rng = rng

        return np.asarray(actions)

    def update(self, batch: Batch) -> InfoDict:
        new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit(
            self.rng, self.actor, self.critic, self.value, self.target_critic,
            batch, self.discount, self.tau, self.expectile, self.beta)
        # the update of actor does not impact the training behavior of value/critic functions.

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        return info

    # def save_ckpt(self, ckpt_folder="ckpt", prefix=""):
    #     save_dir = os.path.join(ckpt_folder, self.name, prefix)
    #
    #     for n_ in self.model_names:
    #         self.__getattribute__(n_).save(os.path.join(save_dir, prefix + n_))
    #         print(f"Successfully save {n_} model to {os.path.join(save_dir, prefix + n_)}")
    #
    # def load_ckpt(self, ckpt_path="ckpt", prefix=""):
    #     save_dir = os.path.join(ckpt_path, self.name, prefix)
    #
    #     for n_ in self.model_names:
    #         model = self.__getattribute__(n_)
    #         self.__setattr__(n_, model.load(os.path.join(save_dir, prefix + n_)))
    #         print(f"Successfully save {n_} model to {os.path.join(save_dir, prefix + n_)}")

        # self.actor = self.actor.load(os.path.join(save_dir, prefix + 'actor'))
        # print(f"Successfully load actor model from {os.path.join(save_dir, prefix + 'actor')}")
        #
        # self.prior_actor = self.prior_actor.load(os.path.join(save_dir, prefix + 'prior_actor'))
        # print(f"Successfully load prior model from {os.path.join(save_dir, prefix + 'prior_actor')}")
