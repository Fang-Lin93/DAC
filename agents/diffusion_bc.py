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
                 clip_sampler: bool = False,
                 time_dim: int = 16,
                 beta_schedule: str = 'vp',
                 lr_decay_steps: int = 2000000,
                 sampler: str = "ddim",
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


if __name__ == '__main__':

    import numpy as np
    from datasets import make_env_and_dataset

    env, dataset = make_env_and_dataset('hopper-medium-expert-v2', 23, 'd4rl')

    learner = DDPMBCLearner(seed=32,
                            observations=env.observation_space.sample()[np.newaxis],
                            actions=env.action_space.sample()[np.newaxis],
                            sampler='ddpm')

    state = env.observation_space.sample()[np.newaxis]
    act_T = env.action_space.sample()[np.newaxis]
    t = jnp.ones((1, 1))

    pred_noise = learner.actor(state, act_T, t)

    batch = dataset.sample(256)

    for i in range(100):
        print(learner.update(batch))

    a_ = learner.sample_actions(observations=env.reset())

    import jax
    import jax.numpy as jnp

    x = jax.random.normal(jax.random.PRNGKey(1), (5, 6, 7))
    x.std()

    #
    #
    #
    # def update_v(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
    #     qs = agent.target_critic.apply_fn(
    #         {"params": agent.target_critic.params},
    #         batch["observations"],
    #         batch["actions"],
    #     )
    #     q = qs.min(axis=0)
    #
    #     def value_loss_fn(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
    #         v = agent.value.apply_fn({"params": value_params}, batch["observations"])
    #
    #         if agent.critic_objective == 'expectile':
    #             value_loss = expectile_loss(q - v, agent.critic_hyperparam).mean()
    #         elif agent.critic_objective == 'quantile':
    #             value_loss = quantile_loss(q - v, agent.critic_hyperparam).mean()
    #         elif agent.critic_objective == 'exponential':
    #             value_loss = exponential_loss(q - v, agent.critic_hyperparam).mean()
    #         else:
    #             raise ValueError(f'Invalid critic objective: {agent.critic_objective}')
    #
    #         return value_loss, {"value_loss": value_loss, "v": v.mean()}
    #
    #     grads, info = jax.grad(value_loss_fn, has_aux=True)(agent.value.params)
    #     value = agent.value.apply_gradients(grads=grads)
    #
    #     agent = agent.replace(value=value)
    #
    #     return agent, info
    #
    # def update_q(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
    #     next_v = agent.value.apply_fn(
    #         {"params": agent.value.params}, batch["next_observations"]
    #     )
    #
    #     target_q = batch["rewards"] + agent.discount * batch["masks"] * next_v
    #     batch_size = batch["observations"].shape[0]
    #
    #     def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
    #         qs = agent.critic.apply_fn(
    #             {"params": critic_params}, batch["observations"], batch["actions"]
    #         )
    #         critic_loss = ((qs - target_q) ** 2).mean()
    #
    #         return critic_loss, {
    #             "critic_loss": critic_loss,
    #             "q": qs.mean(),
    #         }
    #
    #     grads, info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
    #     critic = agent.critic.apply_gradients(grads=grads)
    #
    #     agent = agent.replace(critic=critic)
    #
    #     target_critic_params = optax.incremental_update(
    #         critic.params, agent.target_critic.params, agent.tau
    #     )
    #     target_critic = agent.target_critic.replace(params=target_critic_params)
    #
    #     new_agent = agent.replace(critic=critic, target_critic=target_critic)
    #     return new_agent, info
    #
    # def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
    #     rng = agent.rng
    #     key, rng = jax.random.split(rng, 2)
    #     time = jax.random.randint(key, (batch['actions'].shape[0],), 0, agent.T)
    #     key, rng = jax.random.split(rng, 2)
    #     noise_sample = jax.random.normal(
    #         key, (batch['actions'].shape[0], agent.act_dim))
    #
    #     alpha_hats = agent.alpha_hats[time]
    #     time = jnp.expand_dims(time, axis=1)
    #     alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
    #     alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
    #     noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample
    #
    #     key, rng = jax.random.split(rng, 2)
    #
    #     qs = agent.target_critic.apply_fn(
    #         {"params": agent.target_critic.params},
    #         batch["observations"],
    #         batch["actions"],
    #     )
    #
    #     q = qs.min(axis=0)
    #
    #     v = agent.value.apply_fn(
    #         {"params": agent.value.params}, batch["observations"]
    #     )
    #
    #     adv = q - v
    #
    #     if agent.actor_objective == "soft_adv":
    #         weights = jnp.where(adv > 0, agent.critic_hyperparam, 1 - agent.critic_hyperparam)
    #     elif agent.actor_objective == "hard_adv":
    #         weights = jnp.where(adv >= (-0.01), 1, 0)
    #     elif agent.actor_objective == "exp_adv":
    #         weights = jnp.exp(adv * agent.policy_temperature)
    #         weights = jnp.minimum(weights, 100)  # clip weights
    #     elif agent.actor_objective == "bc":
    #         weights = jnp.ones(adv.shape)
    #     else:
    #         raise ValueError(f'Invalid actor objective: {agent.actor_objective}')
    #
    #     def actor_loss_fn(
    #             score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
    #         eps_pred = agent.score_model.apply_fn({'params': score_model_params},
    #                                               batch['observations'],
    #                                               noisy_actions,
    #                                               time,
    #                                               rngs={'dropout': key},
    #                                               training=True)
    #
    #         actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis=-1) * weights).mean()
    #
    #         return actor_loss, {'actor_loss': actor_loss, 'weights': weights.mean()}
    #
    #     grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
    #     score_model = agent.score_model.apply_gradients(grads=grads)
    #
    #     agent = agent.replace(score_model=score_model)
    #
    #     target_score_params = optax.incremental_update(
    #         score_model.params, agent.target_score_model.params, agent.actor_tau
    #     )
    #
    #     target_score_model = agent.target_score_model.replace(params=target_score_params)
    #
    #     new_agent = agent.replace(score_model=score_model, target_score_model=target_score_model, rng=rng)
    #
    #     return new_agent, info
    #
    # def eval_actions(self, observations: jnp.ndarray):
    #     rng = self.rng
    #
    #     assert len(observations.shape) == 1
    #     observations = jax.device_put(observations)
    #     observations = jnp.expand_dims(observations, axis=0).repeat(self.N, axis=0)
    #
    #     score_params = self.target_score_model.params
    #     actions, rng = ddpm_sampler(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations,
    #                                 self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M,
    #                                 self.clip_sampler)
    #     rng, key = jax.random.split(rng, 2)
    #     qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations, actions)
    #     idx = jnp.argmax(qs)
    #     action = actions[idx]
    #     new_rng = rng
    #
    #     return np.array(action.squeeze()), self.replace(rng=new_rng)
    #
    # def sample_implicit_policy(self, observations: jnp.ndarray):
    #     rng = self.rng
    #
    #     assert len(observations.shape) == 1
    #     observations = jax.device_put(observations)
    #     observations = jnp.expand_dims(observations, axis=0).repeat(self.N, axis=0)
    #
    #     score_params = self.target_score_model.params
    #     actions, rng = ddpm_sampler(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations,
    #                                 self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M,
    #                                 self.clip_sampler)
    #     rng, key = jax.random.split(rng, 2)
    #     qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations, actions)
    #     vs = compute_v(self.value.apply_fn, self.value.params, observations)
    #     adv = qs - vs
    #
    #     if self.critic_objective == 'expectile':
    #         tau_weights = jnp.where(adv > 0, self.critic_hyperparam, 1 - self.critic_hyperparam)
    #         sample_idx = jax.random.choice(key, self.N, p=tau_weights / tau_weights.sum())
    #         action = actions[sample_idx]
    #     elif self.critic_objective == 'quantile':
    #         tau_weights = jnp.where(adv > 0, self.critic_hyperparam, 1 - self.critic_hyperparam)
    #         tau_weights = tau_weights / adv
    #         sample_idx = jax.random.choice(key, self.N, p=tau_weights / tau_weights.sum())
    #         action = actions[sample_idx]
    #     elif self.critic_objective == 'exponential':
    #         weights = self.critic_hyperparam * jnp.abs(adv * self.critic_hyperparam) / jnp.abs(adv)
    #         sample_idx = jax.random.choice(key, self.N, p=weights)
    #         action = actions[sample_idx]
    #     else:
    #         raise ValueError(f'Invalid critic objective: {self.critic_objective}')
    #
    #     new_rng = rng
    #
    #     return np.array(action.squeeze()), self.replace(rng=new_rng)
    #
    # def actor_loss_no_grad(agent, batch: DatasetDict):
    #     rng = agent.rng
    #     key, rng = jax.random.split(rng, 2)
    #     time = jax.random.randint(key, (batch['actions'].shape[0],), 0, agent.T)
    #     key, rng = jax.random.split(rng, 2)
    #     noise_sample = jax.random.normal(
    #         key, (batch['actions'].shape[0], agent.act_dim))
    #
    #     alpha_hats = agent.alpha_hats[time]
    #     time = jnp.expand_dims(time, axis=1)
    #     alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
    #     alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
    #     noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample
    #
    #     key, rng = jax.random.split(rng, 2)
    #
    #     def actor_loss_fn(score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
    #         eps_pred = agent.score_model.apply_fn({'params': score_model_params},
    #                                               batch['observations'],
    #                                               noisy_actions,
    #                                               time,
    #                                               training=False)
    #
    #         actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis=-1)).mean()
    #
    #         return actor_loss, {'actor_loss': actor_loss}
    #
    #     _, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
    #     new_agent = agent.replace(rng=rng)
    #
    #     return new_agent, info
    #
    # @jax.jit
    # def actor_update(self, batch: DatasetDict):
    #     new_agent = self
    #     new_agent, actor_info = new_agent.update_actor(batch)
    #     return new_agent, actor_info
    #
    # @jax.jit
    # def eval_loss(self, batch: DatasetDict):
    #     new_agent = self
    #     new_agent, actor_info = new_agent.actor_loss_no_grad(batch)
    #     return new_agent, actor_info
    #
    # @jax.jit
    # def critic_update(self, batch: DatasetDict):
    #     def slice(x):
    #         return x[:256]
    #
    #     new_agent = self
    #
    #     mini_batch = jax.tree_util.tree_map(slice, batch)
    #     new_agent, critic_info = new_agent.update_v(mini_batch)
    #     new_agent, value_info = new_agent.update_q(mini_batch)
    #
    #     return new_agent, {**critic_info, **value_info}
    #
    # @jax.jit
    # def update(self, batch: DatasetDict):
    #     new_agent = self
    #     batch_size = batch['observations'].shape[0]
    #
    #     def first_half(x):
    #         return x[:batch_size]
    #
    #     def second_half(x):
    #         return x[batch_size:]
    #
    #     first_batch = jax.tree_util.tree_map(first_half, batch)
    #     second_batch = jax.tree_util.tree_map(second_half, batch)
    #
    #     new_agent, _ = new_agent.update_actor(first_batch)
    #     new_agent, actor_info = new_agent.update_actor(second_batch)
    #
    #     def slice(x):
    #         return x[:256]
    #
    #     mini_batch = jax.tree_util.tree_map(slice, batch)
    #     new_agent, critic_info = new_agent.update_v(mini_batch)
    #     new_agent, value_info = new_agent.update_q(mini_batch)
    #
    #     return new_agent, {**actor_info, **critic_info, **value_info}
