
import jax
from typing import Tuple
import jax.numpy as jnp

from networks.types import Batch, InfoDict, Params, PRNGKey
from networks.model import Model


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)
    return target_critic.replace(params=new_target_params)


def update_actor(key: PRNGKey, actor: Model, critic: Model, value: Model,
                 batch: Batch, alpha: float, alg: str) -> Tuple[Model, InfoDict]:
    v = value(batch.observations)
    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)

    if alg == 'sql':
        weight = q - v
        weight = jnp.maximum(weight, 0)
    elif alg == 'eql':
        weight = jnp.exp(10 * (q - v) / alpha)

    else:
        raise NotImplementedError('please choose sql or eql')

    weight = jnp.clip(weight, 0, 100.)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply(actor_params,
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -(weight * log_probs).mean()
        return actor_loss, {'actor_loss': actor_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info


def update_v(critic: Model, value: Model, batch: Batch,
             alpha: float, alg: str) -> Tuple[Model, InfoDict]:

    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply(value_params, batch.observations)
        if alg == 'sql':
            sp_term = (q - v) / (2 * alpha) + 1.0
            sp_weight = jnp.where(sp_term > 0, 1., 0.)
            value_loss = (sp_weight * (sp_term**2) + v / alpha).mean()
        elif alg == 'eql':
            # here it normalizes the weights
            sp_term = (q - v) / alpha
            sp_term = jnp.minimum(sp_term, 5.0)
            max_sp_term = jnp.max(sp_term, axis=0)
            max_sp_term = jnp.where(max_sp_term < -1.0, -1.0, max_sp_term)
            max_sp_term = jax.lax.stop_gradient(max_sp_term)
            value_loss = (jnp.exp(sp_term - max_sp_term) + jnp.exp(-max_sp_term) * v / alpha).mean()
        else:
            raise NotImplementedError('please choose sql or eql')
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
            'q-v': (q - v).mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info


def update_q(critic: Model, value: Model,
             batch: Batch,  discount: float) -> Tuple[Model, InfoDict]:
    next_v = value(batch.next_observations)
    target_q = batch.rewards + discount * batch.masks * next_v
    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply(critic_params, batch.observations,
                              batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


@jax.jit
def _update_jit_sql(
        rng: PRNGKey, actor: Model, critic: Model,
        value: Model, target_critic: Model, batch: Batch, discount: float, tau: float,
        alpha: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:
    new_value, value_info = update_v(target_critic, value, batch, alpha, alg='sql')
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, target_critic,
                                         new_value, batch, alpha, alg='sql')
    new_critic, critic_info = update_q(critic, new_value, batch, discount)

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info
    }


@jax.jit
def _update_jit_eql(
        rng: PRNGKey, actor: Model, critic: Model,
        value: Model, target_critic: Model, batch: Batch, discount: float, tau: float,
        alpha: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:
    new_value, value_info = update_v(target_critic, value, batch, alpha, alg='eql')
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, target_critic,
                                         new_value, batch, alpha, alg='eql')
    new_critic, critic_info = update_q(critic, new_value, batch, discount)

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info
    }