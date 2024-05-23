#  Prior controlled diffusers Q learning

import os
from functools import partial
from typing import Optional, Sequence, Tuple, Union, Callable
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import numpy as np
from networks.model import Model, get_weight_decay_mask
from networks.policies import NormalTanhPolicy, sample_actions
from networks.critics import EnsembleQ
from networks.mlp import MLP
from networks.updates import ema_update
from diffusions.diffusion import DDPM, ddpm_sampler, ddim_sampler
from diffusions.utils import FourierFeatures, cosine_beta_schedule, vp_beta_schedule
from datasets import Batch
from networks.types import InfoDict, Params, PRNGKey
from agents.base import Agent

"""Implementations of algorithms for continuous control."""

EPS = 1e-5


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


@partial(jax.jit,
         static_argnames=('action_decoder', 'num_samples', 'act_dim', 'eta', 'tau', 'need_ema'))
def jit_update_actor(actor: Model,
                     actor_tar: Model,
                     critic: Model,
                     action_decoder: Callable,
                     batch: Batch,
                     rng: PRNGKey,
                     T,
                     alpha_hat,
                     num_samples: int,
                     act_dim: int,
                     eta: float,
                     tau: float,
                     need_ema: bool) -> Tuple[PRNGKey, Model, Model, InfoDict]:
    rng, t_key, noise_key, prior_key = jax.random.split(rng, 4)
    batch_size = batch.observations.shape[0]

    t = jax.random.randint(t_key, (batch_size,), 1, T + 1)[:, jnp.newaxis]
    eps_sample = jax.random.normal(noise_key, batch.actions.shape)
    alpha_1, alpha_2 = jnp.sqrt(alpha_hat[t]), jnp.sqrt(1 - alpha_hat[t])

    noisy_actions = alpha_1 * batch.actions + alpha_2 * eps_sample

    obs = batch.observations.repeat(num_samples, axis=0)  # (B*repeat, obs_dim)
    prior = jax.random.normal(prior_key, (obs.shape[0], act_dim))

    rng, tr_key1, tr_key2, tr_key3 = jax.random.split(rng, 4)

    def actor_loss_fn(actor_paras: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pred_eps = actor.apply(actor_paras,
                               batch.observations,
                               noisy_actions,
                               t,  # t \in range(1, T+1)
                               rngs={'dropout': tr_key1},
                               training=True)
        # behavior cloning loss
        bc_loss = ((pred_eps - eps_sample) ** 2).mean()

        # denoised guidance
        actions, _ = action_decoder(tr_key2, actor.apply, actor_paras, obs, prior)
        q1, q2 = critic(obs, actions).reshape(2, batch_size, num_samples)  # (B, number_samples)

        q_loss1 = - q1.mean()/jax.lax.stop_gradient(jnp.abs(q2).mean()+EPS)
        q_loss2 = - q2.mean()/jax.lax.stop_gradient(jnp.abs(q1).mean()+EPS)

        rand_ = jax.random.uniform(tr_key3)
        q_loss = (rand_ > 0.5) * q_loss1 + (rand_ <= 0.5) * q_loss2

        actor_loss = eta * bc_loss + q_loss
        return actor_loss, {'actor_loss': actor_loss,
                            'bc_loss': bc_loss,
                            'q_loss': q_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    new_actor_tar = ema_update(new_actor, actor_tar, tau) if need_ema else actor_tar
    return rng, new_actor, new_actor_tar, info
    # return rng, actor.apply_gradient(actor_loss_fn)


@partial(jax.jit, static_argnames=('act_dim', 'discount', 'tau', 'action_decoder', 'num_samples', 'maxQ'))
def _jit_update_critic(rng: PRNGKey,
                       critic: Model,
                       critic_tar: Model,
                       actor_tar: Model,
                       act_dim: int,
                       discount: float,
                       tau: float,
                       action_decoder: Callable,
                       batch: Batch,
                       num_samples: int = 10,
                       maxQ: bool = False,
                       ):
    rng, key = jax.random.split(rng)
    batch_size = batch.observations.shape[0]
    input_size = batch_size * num_samples

    # create conditioned obs.
    next_obs = batch.next_observations.repeat(num_samples, axis=0)  # (B*repeat, obs_dim)

    # find the Q-target
    prior = jax.random.normal(key, (input_size, act_dim))
    next_actions, rng = action_decoder(rng, actor_tar.apply, actor_tar.params, next_obs, prior)  # (B*repeat, act_dim)
    next_v = critic_tar(next_obs, next_actions).reshape(batch_size, -1)  # (B, num_samples)
    if maxQ:
        next_v = next_v.max(axis=-1)  # off-policy to estimate optimal Q-function
    else:
        next_v = next_v.mean(axis=-1)  # on-policy to do policy evaluation

    next_v = next_v.min(axis=0)

    target_q = batch.rewards + discount * batch.masks * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply(critic_params, batch.observations,
                              batch.actions)

        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    new_target_critic = ema_update(new_critic, critic_tar, tau)

    return rng, new_critic, new_target_critic, info


@partial(jax.jit, static_argnames=('action_decoder', 'act_dim', 'batch_act', 'num_samples'))
def _jit_sample_actions(rng: PRNGKey,
                        diffusion_model: Model,
                        critic_tar: Model,
                        prior: jnp.ndarray,
                        observations: jnp.ndarray,
                        act_dim: int,
                        action_decoder: Callable,
                        batch_act: bool,
                        num_samples: int,
                        temperature: float) -> [PRNGKey, jnp.ndarray]:
    actions, rng = action_decoder(rng, diffusion_model.apply, diffusion_model.params, observations, prior)

    # eval actions
    qs = critic_tar(observations, actions).min(axis=0).reshape(-1, num_samples)
    # selected_indices = qs.argmax(axis=-1)

    rng, key = jax.random.split(rng)
    selected_indices = jax.random.categorical(key, logits=qs/(1e-4 + temperature), axis=-1)  # softmax
    actions = actions.reshape(-1, num_samples, act_dim)[jnp.arange(len(selected_indices)), selected_indices]

    if batch_act:
        return rng, actions

    # used for evaluation
    return rng, actions[0]


class DQLLearner(Agent):
    # Diffusion policy iteration for offline reinforcement learning
    # set parameters same as DiffusionQL
    # prior controlled/guided diffuser
    name = "dql"
    model_names = ["actor", "actor_tar", "critic", "critic_tar"]

    def __init__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 seed: int,
                 actor_lr: Union[float, optax.Schedule] = 3e-4,
                 critic_lr: Union[float, optax.Schedule] = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256, 256),
                 clip_grad_norm: float = 1,
                 dropout_rate: Optional[float] = None,
                 layer_norm: bool = False,
                 discount: float = 0.99,
                 tau: float = 0.005,  # ema for critic learning
                 update_ema_every: int = 5,
                 step_start_ema: int = 1000,
                 eta: float = 1.,  # balance bc loss and pi loss
                 T: int = 5,  # number of backward steps
                 ddim_step: int = 5,
                 num_q_samples: int = 10,  # number of sampled x_0 for Q updates
                 num_action_samples: int = 50,  # number of sampled actions to select from for action
                 num_last_repeats: int = 0,
                 clip_sampler: bool = True,
                 time_dim: int = 16,
                 beta_schedule: str = 'vp',
                 lr_decay_steps: int = 2000000,
                 sampler: str = "ddim",
                 action_prior: str = 'normal',
                 temperature: float = 1.,
                 actor_path: str = None,
                 maxQ: bool = False,  # whether take maxQ update?
                 **kwargs,
                 ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)
        act_dim = actions.shape[-1]

        # define behavior diffusion model
        time_embedding = partial(FourierFeatures,
                                 output_size=time_dim,
                                 learnable=False)

        time_processor = partial(MLP,
                                 hidden_dims=(32, 32),
                                 activations=mish,
                                 activate_final=False)

        if lr_decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, lr_decay_steps)
            critic_lr = optax.cosine_decay_schedule(critic_lr, lr_decay_steps)

        noise_model = partial(MLP,
                              hidden_dims=tuple(list(hidden_dims) + [act_dim]),
                              activations=mish,
                              layer_norm=layer_norm,
                              dropout_rate=dropout_rate,
                              activate_final=False)

        actor_def = DDPM(time_embedding=time_embedding,
                         time_processor=time_processor,
                         noise_predictor=noise_model)

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations, actions, jnp.zeros((1, 1))],  # time
                             optimizer=optax.adam(learning_rate=actor_lr),
                             clip_grad_norm=clip_grad_norm)
        actor_tar = Model.create(actor_def,
                                 inputs=[actor_key, observations, actions, jnp.zeros((1, 1))])

        critic_def = EnsembleQ(hidden_dims, activations=mish, num_qs=2)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              optimizer=optax.adam(learning_rate=critic_lr),
                              clip_grad_norm=clip_grad_norm)
        critic_tar = Model.create(critic_def,
                                  inputs=[critic_key, observations, actions])

        # models
        self.actor = actor
        self.actor_tar = actor_tar
        self.critic = critic
        self.critic_tar = critic_tar

        if actor_path is not None:
            self.load_actor(actor_path)

        # sampler
        self.action_prior = action_prior
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
        # add a special beginning beta[0] = 0 so that alpha[0] = 1, it is used for ddim
        self.betas = jnp.concatenate([jnp.zeros((1,)), self.betas])

        alphas = 1 - self.betas
        self.alphas = alphas
        self.alpha_hats = jnp.cumprod(alphas)
        self.sqrt_alpha_hat_T = jnp.sqrt(self.alpha_hats[-1])

        assert T >= ddim_step, f"timestep {T} should >= ddim_step {ddim_step}"
        self.T = T
        self.ddim_step = ddim_step

        self.num_q_samples = num_q_samples
        self.num_action_samples = num_action_samples
        self.num_last_repeats = num_last_repeats
        self.act_dim = act_dim
        self.clip_sampler = clip_sampler
        self.discount = discount
        self.tau = tau
        self.maxQ = maxQ

        self.eta = eta
        self.etas = jnp.linspace(0, 0.9, lr_decay_steps)

        # training
        self.rng = rng
        self._n_training_steps = 0
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema

    def eta_schedule(self):
        if self._n_training_steps < len(self.etas):
            return self.etas[self._n_training_steps]
        return 0.9

    def load_actor(self, actor_path):
        self.actor = self.actor.load(actor_path)
        self.actor_tar = self.actor_tar.load(actor_path)
        print(f"Successfully load pre-trained dbc from {actor_path}")

    def action_decoder(self, key: PRNGKey, model_apply_fn: Callable, params: Params, obs: jnp.array, prior: jnp.array):
        if self.sampler == 'ddim':
            return ddim_sampler(key, model_apply_fn, params, self.T, obs, self.alphas,
                                self.alpha_hats, self.temperature, self.num_last_repeats, self.clip_sampler, prior,
                                self.ddim_step)

        elif self.sampler == 'ddpm':
            return ddpm_sampler(key, model_apply_fn, params, self.T, obs, self.alphas,
                                self.alpha_hats, self.temperature, self.num_last_repeats, self.clip_sampler, prior)

        else:
            raise ValueError(f"{self.sampler} not in ['ddpm', 'ddim']")

    def update(self, batch: Batch) -> InfoDict:

        info = {}
        # update conditional diffusion bc
        actor_need_ema = ((self._n_training_steps > self.step_start_ema) and
                          (self._n_training_steps % self.update_ema_every == 0))
        self.rng, self.actor, self.actor_tar, new_info = jit_update_actor(self.actor,
                                                                          self.actor_tar,
                                                                          self.critic,
                                                                          self.action_decoder,
                                                                          batch,
                                                                          self.rng,
                                                                          self.T,
                                                                          self.alpha_hats,
                                                                          self.num_q_samples,
                                                                          self.act_dim,
                                                                          self.eta,
                                                                          self.tau,
                                                                          need_ema=actor_need_ema)  # small stuck

        info.update(new_info)
        # Q-learning
        # self.rng, key = jax.random.split(self.rng)
        self.rng, self.critic, self.critic_tar, new_info = _jit_update_critic(self.rng,
                                                                              self.critic,
                                                                              self.critic_tar,
                                                                              self.actor_tar,
                                                                              self.act_dim,
                                                                              self.discount,
                                                                              self.tau,  # ema
                                                                              self.action_decoder,
                                                                              batch,
                                                                              self.num_q_samples,
                                                                              self.maxQ)
        info.update(new_info)
        self._n_training_steps += 1
        # print(self._n_training_steps)
        return info

    def sample_actions(self,
                       observations: jnp.ndarray,
                       temperature=None,
                       batch_act=False) -> jnp.ndarray:

        # set num_repeats = 1, it's just conditional generation of diffusions
        if len(observations.shape) == 1:
            observations = observations[jnp.newaxis, :]  # batch of actions  (B, dim(obs))

        observations = jax.device_put(observations)
        observations = observations.repeat(self.num_action_samples, axis=0)  # (B*num_samples, dim_obs)

        self.rng, key = jax.random.split(self.rng)
        if self.action_prior == 'zeros':
            prior = jnp.zeros((observations.shape[0], self.act_dim))
        elif self.action_prior == 'normal':
            prior = jax.random.normal(key, (observations.shape[0], self.act_dim))
        else:
            raise ValueError(f"self.action_prior={self.action_prior} not implemented")

        self.rng, action = _jit_sample_actions(self.rng,
                                               self.actor,
                                               self.critic_tar,
                                               prior,
                                               observations,
                                               self.act_dim,
                                               self.action_decoder,
                                               batch_act=batch_act,
                                               num_samples=self.num_action_samples,
                                               temperature=1.)

        return np.asarray(action)
