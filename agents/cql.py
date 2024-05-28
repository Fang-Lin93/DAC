import jax
import optax
import flax.linen as nn
import numpy as np
import jax.numpy as jnp
import functools
from networks.model import Model
from networks.policies import NormalTanhPolicy, sample_actions
from networks.critics import EnsembleQ
from agents.base import Agent
from networks.types import PRNGKey, Params, Tuple, Optional, Batch
from networks.updates import ema_update


# from models import Temperature
# from updates import _update_jit


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)


def update_temperature(temp: Model, log_probs: float,
                       target_entropy: float) -> Tuple[Model, dict]:
    def temperature_loss_fn(temp_params):
        temperature = temp.apply(temp_params)
        temp_loss = - temperature * (log_probs + target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}

    new_temp, info = temp.apply_gradient(temperature_loss_fn)

    return new_temp, info


def update_actor(key: PRNGKey, actor: Model, critic: Model, temp: float,
                 batch: Batch) -> Tuple[Model, dict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, dict]:
        dist = actor.apply(actor_params,
                           batch.observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        qs = critic(batch.observations, actions)
        q = qs.min(axis=0)
        actor_loss = (log_probs * temp - q).mean()
        # actor_loss = (log_probs * temp() - q).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'log_probs': log_probs.mean()
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info


def update_critic(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
                  temp: float, batch: Batch, discount: float,
                  backup_entropy: bool, rand_act_num: int = 10, alpha: float = 1.):
    key, act_key = jax.random.split(key)

    # temperature = temp()
    repeat_obs = batch.observations.repeat(rand_act_num, axis=0)  # (B*repeat, obs_dim)
    rand_acts = jax.random.uniform(act_key, (batch.actions.shape[0] * rand_act_num, batch.actions.shape[1])
                                   , minval=-1., maxval=1.)

    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)  # the a' is sampled ~ pi(.|s')
    next_log_probs = dist.log_prob(next_actions)
    next_qs = target_critic(batch.next_observations, next_actions)

    next_q = next_qs.min(axis=0)  # prevent concrete value operator

    target_q = batch.rewards + discount * batch.masks * next_q

    if backup_entropy:  # down-weight low entropy policy
        target_q -= discount * batch.masks * temp * next_log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, dict]:
        qs = critic.apply(critic_params,
                          batch.observations,
                          batch.actions)  # (H, B)

        # simple CQL
        next_a_qs = critic.apply(critic_params,
                                 batch.observations,
                                 next_actions)  # (H, B)
        next_policy_qs = critic.apply(critic_params,
                                      batch.next_observations,
                                      next_actions)  # (H, B)
        rand_qs = critic.apply(critic_params,
                               repeat_obs,
                               rand_acts)  # (H, B*num_samples)

        all_qs = jnp.concatenate([qs, next_a_qs, next_policy_qs, rand_qs], axis=-1)  # (H, (3+num_samples)*B)
        gap_loss = (jax.nn.logsumexp(all_qs / temp, axis=-1) * temp - qs.mean(axis=-1)).mean()
        td_loss = ((qs - target_q) ** 2).mean()
        critic_loss = td_loss + alpha * gap_loss
        return critic_loss, {
            'critic_loss': critic_loss,
            'gap_loss': gap_loss,
            'qs': qs.mean(),
            'qs_std': qs.std(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info


@functools.partial(jax.jit, static_argnames=('temp', 'backup_entropy', 'rand_act_num', 'alpha'))
def _update_jit(
        rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
        temp: float, batch: Batch, discount: float, tau: float, backup_entropy: bool, rand_act_num: int, alpha: float) \
        -> Tuple[PRNGKey, Model, Model, Model, dict]:
    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            backup_entropy=backup_entropy,
                                            rand_act_num=rand_act_num,
                                            alpha=alpha)

    new_target_critic = ema_update(new_critic, target_critic, tau)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    # new_temp, alpha_info = update_temperature(temp, actor_info['log_probs'],
    #                                           target_entropy)

    return rng, new_actor, new_critic, new_target_critic, {
        **critic_info,
        **actor_info,
        # **alpha_info
    }


class CQLLearner(Agent):
    name = 'cql'
    model_names = ["actor", "critic", "tar_critic"]

    def __init__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 seed: int,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 # temp_lr: float = 3e-4,
                 hidden_dims: Tuple[int] = (256, 256, 256),
                 init_temperature: float = 0.5,
                 discount: float = 0.99,
                 backup_entropy: bool = True,
                 # target_entropy: Optional[float] = None,
                 rand_act_num: int = 10,
                 alpha: float = 1.,  # for weights CQL gap
                 dropout_rate: float = 0.,
                 tau: float = 0.005,  # used for EMA
                 lr_decay_steps: Optional[int] = None,
                 opt_decay_schedule: str = "cosine",
                 num_qs: int = 2,
                 # rem: bool = False,  # random ensemble mixture of Q values
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
                                     tanh_squash_distribution=False)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, lr_decay_steps)
            act_opt = optax.chain(optax.scale_by_adam(),
                                  optax.scale_by_schedule(schedule_fn))
        else:
            act_opt = optax.adam(learning_rate=actor_lr)

        self.actor = Model.create(actor_net,
                                  inputs=[actor_key, observations],
                                  optimizer=act_opt)

        self.critic = Model.create(EnsembleQ(hidden_dims, num_qs=num_qs),
                                   inputs=[critic_key, observations, actions],
                                   optimizer=optax.adam(learning_rate=critic_lr))

        self.tar_critic = Model.create(EnsembleQ(hidden_dims, num_qs=num_qs),
                                       inputs=[critic_key, observations, actions],
                                       optimizer=optax.adam(learning_rate=critic_lr))

        # self.temp = Model.create(Temperature(init_temperature),
        #                          inputs=[temp_key],
        #                          optimizer=optax.adam(learning_rate=temp_lr))

        self.temp = init_temperature
        self.discount = discount
        self.backup_entropy = backup_entropy
        self.rand_act_num = rand_act_num
        self.alpha = alpha
        # if target_entropy is None:
        #     self.target_entropy = action_dim / 2
        # else:
        #     self.target_entropy = target_entropy
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
        new_rng, new_actor, new_critic, new_target_critic, info = _update_jit(
            self.rng, self.actor, self.critic, self.tar_critic, self.temp,
            batch, self.discount, self.tau, self.backup_entropy, self.rand_act_num, self.alpha)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.tar_critic = new_target_critic
        # self.temp = new_temp
        self.update_step += 1

        return info
