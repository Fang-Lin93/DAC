"""Implementations of algorithms for continuous control."""
from functools import partial
from typing import Optional, Sequence, Tuple, Union
import flax.linen as nn
import jax
import os
import jax.numpy as jnp
import optax
import numpy as np
from networks.model import Model, get_weight_decay_mask
from networks.mlp import MLP
from diffusions.diffusion import DDPM, ddpm_sampler, ddim_sampler, jit_update_diffusion_model
from diffusions.utils import FourierFeatures, cosine_beta_schedule, vp_beta_schedule
from networks.types import InfoDict, Params, PRNGKey, Batch
from agents.base import Agent


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


class DDPMBCLearner(Agent):
    # set parameters same as DiffusionQL
    name = "dbc"
    model_names = ["actor"]

    def __init__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 seed: int,
                 actor_lr: Union[float, optax.Schedule] = 3e-4,
                 clip_grad_norm: float = 1,
                 hidden_dims: Sequence[int] = (256, 256, 256),
                 dropout_rate: Optional[float] = None,
                 layer_norm: bool = False,
                 T: int = 100,  # number of backward steps
                 ddim_step: int = 5,
                 num_samples: int = 1,  # number of sampled x_0
                 num_last_repeats: int = 0,
                 clip_sampler: bool = True,
                 time_dim: int = 16,
                 beta_schedule: str = 'vp',
                 lr_decay_steps: int = 2000000,
                 sampler: str = "ddpm",
                 temperature: float = 1,
                 **kwargs
                 ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng)

        act_dim = actions.shape[-1]

        time_embedding = partial(FourierFeatures,
                                 output_size=time_dim,
                                 learnable=False)

        time_processor = partial(MLP,
                                 hidden_dims=(32, 32),
                                 activations=mish,
                                 activate_final=False)

        if lr_decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, lr_decay_steps)

        noise_model = partial(MLP,
                              hidden_dims=tuple(list(hidden_dims) + [act_dim]),
                              activations=mish,
                              layer_norm=layer_norm,
                              dropout_rate=dropout_rate,
                              activate_final=False)

        actor_def = DDPM(time_embedding=time_embedding,
                         time_processor=time_processor,
                         noise_predictor=noise_model)

        # optimizer = optax.adamw(learning_rate=actor_lr,
        #                         mask=get_weight_decay_mask)
        self.actor = Model.create(actor_def,  # state, action_t, timestep -> action_(t-1) || first dim=batch
                                  inputs=[actor_key, observations, actions, jnp.zeros((1, 1))],
                                  optimizer=optax.adam(learning_rate=actor_lr),
                                  clip_grad_norm=clip_grad_norm)

        self.sampler = sampler
        self.temperature = temperature

        if beta_schedule == 'cosine':
            self.betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == 'linear':
            self.betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == 'vp':
            self.betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')

        self.betas = jnp.concatenate([jnp.zeros((1,)), self.betas])
        # add a special beginning beta[0] = 0 so that alpha[0] = 1, it is used for ddim

        alphas = 1 - self.betas
        self.alphas = alphas
        self.alpha_hats = jnp.cumprod(alphas)
        self.T = T
        self.ddim_step = ddim_step
        # c = T // ddim_step  # jump step
        # self.ddim_time_seq = jnp.concatenate([jnp.arange(T, 0, -c), jnp.array([0])])
        self.num_samples = num_samples
        self.num_last_repeats = num_last_repeats
        self.act_dim = act_dim
        self.clip_sampler = clip_sampler
        self.rng = rng

        self._n_training_steps = 0

    def update(self, batch: Batch) -> InfoDict:
        self.rng, (self.actor, info) = jit_update_diffusion_model(self.actor, batch, self.rng, self.T, self.alpha_hats)

        self._n_training_steps += 1
        return info

    def sample_actions(self,
                       observations: jnp.ndarray,
                       temperature: float = 0,
                       batch_act: bool = False) -> jnp.ndarray:
        rng = self.rng

        if len(observations.shape) == 1:
            observations = observations[jnp.newaxis, :]
        observations = jax.device_put(observations)
        observations = observations.repeat(self.num_samples, axis=0)

        rng, key = jax.random.split(rng, 2)
        prior = jax.random.normal(key, (observations.shape[0], self.act_dim))  # (batch_size, act_dim)

        # actions, self.rng = self.sampler(self.actor.apply, self.actor.params, self.T, rng, observations,
        #                                  self.alphas, self.alpha_hats, self.betas, temperature, self.num_last_repeats,
        #                                  self.clip_sampler, prior, False)

        if self.sampler == 'ddpm':
            actions, self.rng = ddpm_sampler(rng, self.actor.apply, self.actor.params, self.T,
                                             observations, self.alphas, self.alpha_hats, self.temperature,
                                             self.num_last_repeats, self.clip_sampler, prior)
        elif self.sampler == 'ddim':
            actions, self.rng = ddim_sampler(key, self.actor.apply, self.actor.params, self.T, observations,
                                             self.alphas, self.alpha_hats, self.temperature, self.num_last_repeats,
                                             self.clip_sampler, prior, self.ddim_step)
        else:
            raise ValueError(f"sampling method {self.sampler} is not in ['ddpm', 'ddim']")

        if batch_act:
            return np.asarray(actions)

        # used for evaluation
        return np.asarray(actions[0])

        # return np.asarray(actions[0]).clip(-1, 1)
