import jax
import jax.numpy as jnp
import functools
from jax import random
from networks.model import Model
from datasets import Batch
from typing import Tuple
from networks.types import PRNGKey, Params


def update_target(critic: Model, target_critic: Model, tau: float) -> Model:
    # EMA of parameters
    new_target_params = jax.jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


def update_temperature(temp: Model, log_probs: float,
                       target_entropy: float) -> Tuple[Model, dict]:
    def temperature_loss_fn(temp_params):
        temperature = temp.apply(temp_params)
        temp_loss = - temperature * (log_probs + target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}

    new_temp, info = temp.apply_gradient(temperature_loss_fn)

    return new_temp, info


def update_actor(key: PRNGKey, actor: Model, critic: Model, temp: Model,
                 batch: Batch) -> Tuple[Model, dict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, dict]:
        dist = actor.apply(actor_params,
                           batch.observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'log_probs': log_probs.mean()
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info


# def update_critic(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
#                   temp: Model, batch: Batch, discount: float,
#                   backup_entropy: bool) -> Tuple[Model, dict]:
#     dist = actor(batch.next_observations)
#     next_actions = dist.sample(seed=key)
#     next_log_probs = dist.log_prob(next_actions)
#
#     next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
#     next_q = jnp.minimum(next_q1, next_q2)
#
#     target_q = batch.rewards + discount * batch.masks * next_q
#
#     if backup_entropy:
#         target_q -= discount * batch.masks * temp() * next_log_probs
#
#     def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, dict]:
#         q1, q2 = critic.apply({'params': critic_params},
#                               batch.observations,
#                               batch.actions)
#         critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
#         return critic_loss, {
#             'critic_loss': critic_loss,
#             'q1': q1.mean(),
#             'q2': q2.mean()
#         }
#
#     new_critic, info = critic.apply_gradient(critic_loss_fn)
#
#     return new_critic, info


def update_critic(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
                  temp: Model, batch: Batch, discount: float,
                  backup_entropy: bool, rand_ensemble_q: bool = False):
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)  # the a' is sampled ~ pi(.|s')
    next_log_probs = dist.log_prob(next_actions)

    next_qs = target_critic(batch.next_observations, next_actions)

    # support random ensemble mixture of Q functions with multi-heads
    if rand_ensemble_q:
        alphas = random.uniform(key, next_qs.shape)  #  if rand_ensemble else jnp.ones_like(target_qs.shape)
        alphas /= (alphas.sum(axis=0)[jnp.newaxis, :] + 1e-5)
        next_qs = (next_qs * alphas).sum(axis=0)
    else:
        alphas = jnp.ones_like(next_qs)  # (n_head, batch_size)
        next_qs = next_qs.min(axis=0)  # prevent concrete value operator
        # next_qs = jnp.minimum(next_qs, axis=0)  # prevent concrete value operator

    target_q = batch.rewards + discount * batch.masks * next_qs

    if backup_entropy:  # down-weight low entropy policy
        target_q -= discount * batch.masks * temp() * next_log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, dict]:
        qs = critic.apply({'params': critic_params},
                          batch.observations,
                          batch.actions)

        qs = (qs * alphas).sum(axis=0)

        critic_loss = ((qs - target_q) ** 2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'qs': qs.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info


@functools.partial(jax.jit, static_argnames=('backup_entropy', 'rand_ensemble_q'))
def _update_jit(
        rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
        temp: Model, batch: Batch, discount: float, tau: float,
        target_entropy: float, backup_entropy: bool, rand_ensemble_q: bool) \
        -> Tuple[PRNGKey, Model, Model, Model, Model, dict]:
    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            backup_entropy=backup_entropy,
                                            rand_ensemble_q=rand_ensemble_q)

    new_target_critic = update_target(new_critic, target_critic, tau)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = update_temperature(temp, actor_info['log_probs'],
                                              target_entropy)

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }
