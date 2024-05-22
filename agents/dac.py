
from functools import partial
from typing import Optional, Sequence, Tuple, Union, Callable
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from networks.model import Model
from networks.critics import EnsembleQ
from networks.mlp import MLP
from networks.resnet import MLPResNet
from networks.updates import ema_update
from diffusions.diffusion import DDPM, ddpm_sampler, ddim_sampler, ddpm_sampler_with_q_guidance
from diffusions.utils import FourierFeatures, cosine_beta_schedule, vp_beta_schedule
from datasets import Batch
from networks.types import InfoDict, Params, PRNGKey
from agents.base import Agent


EPS = 1e-6


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


@partial(jax.jit, static_argnames=('T', 'Q_guidance', 'use_guidance_loss', 'ema_tau', 'need_ema',
                                   'eta_min', 'eta_max', 'eta_lr', 'bc_threshold', 'action_decoder'))
def jit_update_actor(rng: PRNGKey,
                     actor: Model,
                     actor_tar: Model,
                     critic_tar: Model,
                     T,
                     alpha_hats,
                     eta,  # learnable, thus not static float
                     eta_min: float,
                     eta_max: float,
                     eta_lr,
                     bc_threshold,
                     Q_guidance: str,
                     use_guidance_loss: bool,
                     ema_tau: float,
                     need_ema: bool,
                     batch: Batch,
                     action_decoder: Callable,  # for denoised action only
                     ) -> Tuple[PRNGKey, float, Model, Model, InfoDict]:
    rng, t_key, noise_key, dropout_key = jax.random.split(rng, 4)
    batch_size = batch.observations.shape[0]

    t = jax.random.randint(t_key, (batch_size,), 1, T + 1)[:, jnp.newaxis]
    eps_sample = jax.random.normal(noise_key, batch.actions.shape)

    noise_level = jnp.sqrt(1 - alpha_hats[t])  # (B,)
    noisy_actions = jnp.sqrt(alpha_hats[t]) * batch.actions + noise_level * eps_sample

    q = critic_tar(batch.observations, batch.actions)
    Q_norm = jnp.abs(q).mean()
    if use_guidance_loss:
        Q_grad_fn = jax.vmap(jax.grad(lambda a, s: critic_tar(s, a).mean(axis=0)))
        Q_grad = Q_grad_fn(noisy_actions, batch.observations) / (Q_norm + EPS)  # (B, dimA)
    else:
        Q_grad = 0

    def actor_loss_fn(actor_paras: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pred_eps = actor.apply(actor_paras,
                               batch.observations,
                               noisy_actions,
                               t,  # t \in range(1, T+1)
                               rngs={'dropout': dropout_key},
                               training=True)

        bc_loss = ((pred_eps - eps_sample) ** 2).mean()
        if Q_guidance == 'soft':
            guidance_loss = (noise_level * Q_grad * pred_eps).mean()
        elif Q_guidance == 'hard':
            guidance_loss = (Q_grad * pred_eps).mean()
        elif Q_guidance == 'denoised':
            prior_key, denoise_key = jax.random.split(noise_key, 2)
            prior = jax.random.normal(prior_key, pred_eps.shape)
            actions, _ = action_decoder(denoise_key, actor.apply, actor_paras, batch.observations, prior,
                                        1, True)  # temperature=1, clip_sampler=True
            qs = critic_tar(batch.observations, actions).mean()  # (H, B) -> scalar
            guidance_loss = - qs / (Q_norm + EPS)
        else:
            raise ValueError(f'Q_guidance={Q_guidance} is not supported')

        actor_loss = eta * bc_loss + guidance_loss

        return actor_loss, {'actor_loss': actor_loss,
                            'bc_loss': bc_loss,
                            'guidance_loss': guidance_loss,
                            'eta': eta,
                            'Q_norm': Q_norm,
                            'Q_grad': Q_grad.mean(),
                            'Q_grad_abs': jnp.abs(Q_grad).mean(),
                            }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    # update eta with dual gradient ascent
    if eta_lr > 0 and use_guidance_loss:
        eps_new = actor(batch.observations,
                        noisy_actions,
                        t,
                        rngs={'dropout': dropout_key},
                        training=False)
        new_bc_loss = ((eps_new - eps_sample) ** 2).mean()
        eta += eta_lr * (new_bc_loss - bc_threshold).clip(-1, 1)
        # eta += eta_lr * (info['bc_loss'] - bc_threshold).clip(-1, 1)
        eta = jnp.clip(eta, eta_min, eta_max)  # larger -> BC; small -> greedy(Q)

    new_actor_tar = ema_update(new_actor, actor_tar, ema_tau) if need_ema else actor_tar
    return rng, eta, new_actor, new_actor_tar, info


@partial(jax.jit, static_argnames=('act_dim', 'discount', 'ema_tau', 'action_decoder', 'rho', 'temperature',
                                   'num_samples', 'clip_sampler', 'q_tar', 'maxQ'))
def _jit_update_critic(rng: PRNGKey,
                       critic: Model,
                       critic_tar: Model,
                       actor_tar: Model,
                       act_dim: int,
                       discount: float,
                       ema_tau: float,
                       action_decoder: Callable,
                       rho: float,
                       temperature: float,
                       num_samples: int,
                       clip_sampler: bool,
                       batch: Batch,
                       q_tar: str,
                       maxQ: bool
                       ):
    batch_size = batch.observations.shape[0]
    input_size = batch_size * num_samples

    # create conditioned obs.
    next_obs = batch.next_observations.repeat(num_samples, axis=0)  # (B*repeat, obs_dim)

    # find the Q-target
    rng, key = jax.random.split(rng, 2)
    prior = jax.random.normal(key, (input_size, act_dim))
    next_actions, rng = action_decoder(rng, actor_tar.apply, actor_tar.params, next_obs, prior,
                                       temperature, clip_sampler)  # (B*rep, a_dim)
    next_v = critic_tar(next_obs, next_actions).reshape(-1, batch_size, num_samples)  # (H, B, N_sample)

    if maxQ:
        next_v = next_v.max(axis=-1)  # used for ant-maze
    else:
        next_v = next_v.mean(axis=-1)
    # when rho is large, it becomes pessimistic
    if q_tar == 'min':  # minimal ensemble
        next_v = next_v.min(axis=0)
    elif q_tar == 'convex':
        next_v = rho * next_v.min(axis=0) + (1-rho) * next_v.max(axis=0)
    elif q_tar == 'lcb':  # lcb over ensemble
        next_v = next_v.mean(axis=0) - rho * next_v.std(axis=0)
    else:
        raise NotImplementedError(f'Unrecognized Q-target type={q_tar}')

    # find the Q-target
    target_q = batch.rewards + discount * batch.masks * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        qs = critic.apply(critic_params, batch.observations,
                          batch.actions)  # (H, B)
        critic_loss = ((qs - target_q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'qs': qs.mean(),
            'qs_std': qs.std(axis=0).mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    new_target_critic = ema_update(new_critic, critic_tar, ema_tau)

    return rng, new_critic, new_target_critic, info


@jax.jit
def _jit_update_v(critic_tar: Model,
                  value: Model,
                  batch: Batch) -> Tuple[Model, InfoDict]:
    q = critic_tar(batch.observations, batch.actions).min(axis=0)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply(value_params, batch.observations)
        value_loss = ((q - v) ** 2).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info


@partial(jax.jit, static_argnames=('act_model_apply_fn', 'critic_tar_apply_fn', 'action_decoder',
                                   'act_dim', 'batch_act', 'num_samples', 'temperature', 'clip_sampler'))
def _jit_sample_actions(rng: PRNGKey,
                        act_model_apply_fn: Callable,
                        act_params: Params,
                        critic_tar_apply_fn: Callable,
                        critic_tar_params: Params,
                        prior: jnp.ndarray,
                        temperature: float,  # control softmax only
                        observations: jnp.ndarray,
                        act_dim: int,
                        action_decoder: Callable,
                        batch_act: bool,
                        num_samples: int,
                        # argmax: bool,
                        clip_sampler: bool) -> [PRNGKey, jnp.ndarray]:
    actions, rng = action_decoder(rng, act_model_apply_fn, act_params, observations, prior, 1, clip_sampler)
    rng, key = jax.random.split(rng)

    # eval actions
    qs = critic_tar_apply_fn(critic_tar_params, observations, actions).mean(axis=0).reshape(-1, num_samples)
    if temperature <= 0:  # deterministic action
        selected_indices = qs.argmax(axis=-1)
    else:
        qs = qs / (jnp.abs(qs) + EPS)
        selected_indices = jax.random.categorical(key, logits=qs / (EPS + temperature), axis=-1)  # softmax
    actions = actions.reshape(-1, num_samples, act_dim)[jnp.arange(qs.shape[0]), selected_indices]

    if batch_act:
        return rng, actions

    # used for evaluation
    return rng, actions[0]


class DACLearner(Agent):
    # Diffusion policy iteration for offline reinforcement learning
    # set most parameters the same as DiffusionQL
    name = "dac"
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
                 ema_tau: float = 0.005,  # ema for critic learning
                 update_ema_every: int = 5,  # ema interval for actor learning
                 step_start_ema: int = 1000,
                 eta: float = 1.,  # balance bc loss and Q gradient loss
                 eta_min: float = 0.001,
                 eta_max: float = 100.,
                 eta_lr: float = 0.,
                 rho: float = 1,  # scale of lcb
                 bc_threshold: float = 0.1,
                 T: int = 5,  # number of backward steps
                 ddim_step: int = 5,
                 num_q_samples: int = 10,  # number of sampled x_0 for Q updates
                 num_action_samples: int = 50,  # number of sampled actions to select from for action
                 Q_guidance: str = "soft",
                 use_guidance_loss: bool = True,
                 act_with_q_guid: bool = False,
                 # action_argmax: bool = True,
                 num_last_repeats: int = 0,
                 clip_sampler: bool = False,
                 time_dim: int = 16,
                 beta_schedule: str = 'vp',
                 lr_decay_steps: int = 2000000,
                 sampler: str = "ddpm",
                 action_prior: Union[str, Callable[[PRNGKey, tuple], jnp.ndarray]] = 'normal',
                 temperature: float = 0.,
                 actor_path: str = None,
                 num_qs: int = 2,  # number of Q in Q-ensemble
                 q_tar: str = 'lcb',
                 maxQ: bool = False,  # whether taking the maximum-Q over actions in critic learning
                 resnet: bool = False,
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

        if resnet:
            noise_model = partial(MLPResNet,
                                  num_blocks=2,
                                  hidden_dim=hidden_dims[0],
                                  out_dim=act_dim,
                                  activations=mish,
                                  layer_norm=True,
                                  dropout_rate=dropout_rate)
        else:
            noise_model = partial(MLP,
                                  hidden_dims=(*hidden_dims, act_dim),
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

        critic_def = EnsembleQ(hidden_dims=hidden_dims,
                               activations=mish,
                               num_qs=num_qs)
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
        # self.value = value

        if actor_path is not None:
            self.load_actor(actor_path)

        # sampler
        self.action_prior = action_prior
        self.sampler = sampler
        self.temperature = temperature
        # self.action_argmax = temperature <= 0

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
        self.act_with_q_guid = act_with_q_guid
        # self.guidance_scale = 1/(eta+EPS)
        self.act_dim = act_dim
        self.clip_sampler = clip_sampler
        self.discount = discount
        self.ema_tau = ema_tau
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.q_tar = q_tar
        self.maxQ = maxQ

        self.eta = jnp.array(eta)
        self.eta_min, self.eta_max = eta_min, eta_max
        self.eta_lr = eta_lr
        self.rho = rho
        self.bc_threshold = bc_threshold
        self.Q_guidance = Q_guidance
        self.use_guidance_loss = use_guidance_loss

        self.rng = rng
        self._n_training_steps = 0

    def load_actor(self, actor_path):
        self.actor = self.actor.load(actor_path)
        self.actor_tar = self.actor_tar.load(actor_path)
        print(f"Successfully load pre-trained dbc from {actor_path}")

    def ddpm_action_decoder(self, key: PRNGKey, model_apply_fn: Callable,
                            params: Params, obs: jnp.array, prior: jnp.array, temperature: float):
        return ddpm_sampler(key, model_apply_fn, params, self.T, obs, self.alphas,
                            self.alpha_hats, temperature, self.num_last_repeats, self.clip_sampler, prior)

    def ddpm_critic_guidance_decoder(self, key: PRNGKey, model_apply_fn: Callable,
                                     params: Params, obs: jnp.array, prior: jnp.array, temperature: float):
        return ddpm_sampler_with_q_guidance(key, model_apply_fn, params, self.critic_tar.apply, self.critic_tar.params,
                                            1/(self.eta + EPS), self.T, obs, self.alphas, self.alpha_hats,
                                            temperature, self.num_last_repeats, self.clip_sampler,
                                            prior)

    def action_decoder(self, key: PRNGKey, model_apply_fn: Callable, params: Params, obs: jnp.array, prior: jnp.array,
                       temperature: float, clip_sampler: bool):
        # the 'self' arguments will be cached and temporal changes are not used!
        if self.sampler == 'ddim':
            return ddim_sampler(key, model_apply_fn, params, self.T, obs, self.alphas,
                                self.alpha_hats, temperature, self.num_last_repeats, clip_sampler, prior,
                                self.ddim_step)

        elif self.sampler == 'ddpm':
            return ddpm_sampler(key, model_apply_fn, params, self.T, obs, self.alphas,
                                self.alpha_hats, temperature, self.num_last_repeats, clip_sampler, prior)

        elif self.sampler == 'ddpm_qg':  # 2-stage q guidance
            return ddpm_sampler_with_q_guidance(key, model_apply_fn, params, self.critic_tar.apply,
                                                self.critic_tar.params,
                                                1/(self.eta + EPS), self.T, obs, self.alphas, self.alpha_hats,
                                                temperature, self.num_last_repeats, clip_sampler,
                                                prior)
        else:
            raise ValueError(f"{self.sampler} not in ['ddpm', 'ddim', 'ddpm_qg]")

    def update(self, batch: Batch) -> InfoDict:

        info = {}
        actor_need_ema = ((self._n_training_steps % self.update_ema_every == 0) and
                          (self._n_training_steps > self.step_start_ema))
        # Q-learning
        self.rng, self.critic, self.critic_tar, q_info = _jit_update_critic(self.rng,
                                                                            self.critic,
                                                                            self.critic_tar,
                                                                            self.actor_tar,
                                                                            self.act_dim,
                                                                            self.discount,
                                                                            self.ema_tau,  # ema
                                                                            self.action_decoder,
                                                                            self.rho,
                                                                            1,  # diffusion noise scale
                                                                            self.num_q_samples,
                                                                            self.clip_sampler,
                                                                            batch,
                                                                            self.q_tar,
                                                                            self.maxQ
                                                                            )
        info.update(q_info)

        self.rng, self.eta, self.actor, self.actor_tar, act_info = jit_update_actor(self.rng,
                                                                                    self.actor,
                                                                                    self.actor_tar,
                                                                                    self.critic_tar,
                                                                                    self.T,
                                                                                    self.alpha_hats,
                                                                                    self.eta,
                                                                                    self.eta_min,
                                                                                    self.eta_max,
                                                                                    self.eta_lr,
                                                                                    self.bc_threshold,
                                                                                    self.Q_guidance,
                                                                                    self.use_guidance_loss,
                                                                                    self.ema_tau,
                                                                                    actor_need_ema,
                                                                                    batch,
                                                                                    self.action_decoder,
                                                                                    )
        info.update(act_info)
        self._n_training_steps += 1
        return info

    def sample_actions(self,
                       observations: jnp.ndarray,
                       temperature=None,
                       batch_act=False,
                       prior_fn: Callable = None
                       ) -> jnp.ndarray:

        if len(observations.shape) == 1:
            observations = observations[jnp.newaxis, :]  # (B, dim(obs))

        observations = jax.device_put(observations)
        observations = observations.repeat(self.num_action_samples, axis=0)  # (B*num_samples, dim(obs))

        self.rng, key = jax.random.split(self.rng)
        if self.action_prior == 'zeros':
            prior = jnp.zeros((observations.shape[0], self.act_dim))
        elif self.action_prior == 'normal':
            prior = jax.random.normal(key, (observations.shape[0], self.act_dim))
        elif callable(self.action_prior):
            prior = self.action_prior(key, (observations.shape[0], self.act_dim))
        else:
            raise NotImplementedError(f"self.action_prior={self.action_prior} not implemented")

        action_decoder = self.ddpm_critic_guidance_decoder if self.act_with_q_guid else self.action_decoder

        self.rng, action = _jit_sample_actions(self.rng,
                                               self.actor.apply,
                                               self.actor.params,
                                               self.critic_tar.apply,
                                               self.critic_tar.params,
                                               prior,
                                               self.temperature,  # sample actions using softmax
                                               observations,
                                               self.act_dim,
                                               action_decoder,
                                               batch_act=batch_act,
                                               num_samples=self.num_action_samples,
                                               # argmax=self.action_argmax,
                                               clip_sampler=self.clip_sampler)
        return action

