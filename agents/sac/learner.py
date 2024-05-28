
import jax
import optax
import numpy as np
import jax.numpy as jnp
from networks.model import Model
from networks.policies import NormalTanhPolicy, sample_actions
from networks.critics import EnsembleQ
from datasets import Batch
from typing import Tuple, Optional
from .models import Temperature
from .updates import _update_jit
from agents.base import Agent


class SACLearner(Agent):

    name = 'sac'
    model_names = ["actor", "critic", "tar_critic"]

    def __init__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 seed: int,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 hidden_dims: Tuple[int] = (256, 256),
                 init_temperature: float = 1.0,
                 discount: float = 0.99,
                 backup_entropy: bool = True,
                 target_entropy: Optional[float] = None,
                 rand_ensemble_q: bool = False,
                 dropout_rate: float = 0.,
                 tau: float = 0.005,  # used for EMA
                 lr_decay_steps: Optional[int] = None,
                 opt_decay_schedule: str = "cosine",
                 layer_norm: bool = True,
                 rem: bool = False,  # random ensemble mixture of Q values
                 **kwargs):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]

        actor_net = NormalTanhPolicy(hidden_dims,
                                     action_dim,
                                     log_std_scale=1e-3,
                                     log_std_min=-5.0,
                                     dropout_rate=dropout_rate,
                                     state_dependent_std=False,
                                     tanh_squash_distribution=False,
                                     layer_norm=layer_norm)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, lr_decay_steps)
            act_opt = optax.chain(optax.scale_by_adam(),
                                  optax.scale_by_schedule(schedule_fn))
        else:
            act_opt = optax.adam(learning_rate=actor_lr)

        self.actor = Model.create(actor_net,
                                  inputs=[actor_key, observations],
                                  optimizer=act_opt)

        self.critic = Model.create(EnsembleQ(hidden_dims),
                                   inputs=[critic_key, observations, actions],
                                   optimizer=optax.adam(learning_rate=critic_lr))

        self.tar_critic = Model.create(EnsembleQ(hidden_dims),
                                       inputs=[critic_key, observations, actions],
                                       optimizer=optax.adam(learning_rate=critic_lr))

        self.temp = Model.create(Temperature(init_temperature),
                                 inputs=[temp_key],
                                 optimizer=optax.adam(learning_rate=temp_lr))

        self.discount = discount
        self.backup_entropy = backup_entropy
        self.rand_ensemble_q = rand_ensemble_q
        if target_entropy is None:
            self.target_entropy = action_dim / 2
        else:
            self.target_entropy = target_entropy
        self.tau = tau

        self.rng = rng

        self.update_step = 0

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = sample_actions(self.rng, self.actor.network,
                                      self.actor.params, observations,
                                      temperature, False)
        self.rng = rng
        return np.asarray(actions)

    def update(self, batch: Batch) -> dict:
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self.rng, self.actor, self.critic, self.tar_critic, self.temp,
            batch, self.discount, self.tau, self.target_entropy,
            self.backup_entropy, self.rand_ensemble_q)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.tar_critic = new_target_critic
        self.temp = new_temp

        self.update_step += 1

        return info
