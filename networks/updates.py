from typing import Tuple
import jax
import jax.numpy as jnp
from datasets import Batch
from networks.model import Model
from networks.types import Params, PRNGKey, InfoDict


def log_prob_update(actor: Model, batch: Batch,
                    rng: PRNGKey) -> Tuple[Model, InfoDict]:
    rng, key = jax.random.split(rng)

    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -log_probs.mean()
        return actor_loss, {'actor_loss': actor_loss}

    return rng, *actor.apply_gradient(loss_fn)


def ema_update(src_model: Model, tar_model: Model, tau: float) -> Model:
    # tau is small
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), src_model.params,
        tar_model.params)
    return tar_model.replace(params=new_target_params)

