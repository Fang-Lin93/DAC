

from typing import Tuple

import jax.numpy as jnp
import jax
from networks.model import Model
from networks.types import InfoDict, Params, PRNGKey, Batch
from networks.updates import ema_update


# advantage weighted regression loss for actor recovery
def awr_update_actor(key: PRNGKey, actor: Model, critic: Model, value: Model,
                     batch: Batch, beta: float) -> Tuple[Model, InfoDict]:

    # Model object defines __call__ function, here means value.network.apply({"params":self.params}, obs)
    v = value(batch.observations)

    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)
    exp_a = jnp.exp((q - v) * beta)
    exp_a = jnp.minimum(exp_a, 100.0)  # truncate the weights...

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # it creates a function of params, so it has to directly use net.apply() function rather than
        # the __call__() function of the Model object "actor"
        # here training is True for this forward pass is used for gradient descent
        dist = actor.apply(actor_params,
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss, {'actor_loss': actor_loss, 'adv': q - v}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info


def ept_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))  # (condition, set, otherwise_set)
    return weight * (diff**2)


def update_v(critic: Model, value: Model, batch: Batch,
             expectile: float) -> Tuple[Model, InfoDict]:
    actions = batch.actions
    q1, q2 = critic(batch.observations, actions)
    q = jnp.minimum(q1, q2)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply(value_params, batch.observations)
        value_loss = ept_loss(q - v, expectile).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info


def update_q(critic: Model, target_value: Model, batch: Batch,
             discount: float) -> Tuple[Model, InfoDict]:
    next_v = target_value(batch.next_observations)

    target_q = batch.rewards + discount * batch.masks * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply(critic_params, batch.observations,
                              batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


# def ema_update(src_model: Model, tar_model: Model, tau: float) -> Model:
#     # tau is small
#     new_target_params = jax.tree_util.tree_map(
#         lambda p, tp: p * tau + tp * (1 - tau), src_model.params,
#         tar_model.params)
#     return tar_model.replace(params=new_target_params)


@jax.jit
def _update_jit(
        rng: PRNGKey, actor: Model, critic: Model, value: Model,
        target_critic: Model, batch: Batch, discount: float, tau: float,
        expectile: float, beta: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:
    # the Model.apply_gradient/Model.replace function gives a new Model object with new params
    new_value, value_info = update_v(target_critic, value, batch, expectile)
    key, rng = jax.random.split(rng)
    new_actor, actor_info = awr_update_actor(key, actor, target_critic,
                                             new_value, batch, beta)

    new_critic, critic_info = update_q(critic, new_value, batch, discount)

    new_target_critic = ema_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info
    }





