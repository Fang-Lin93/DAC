import os
import re
from abc import ABC

import flax
import flax.linen as nn
from flax import struct
import jax
import jax.numpy as jnp
import optax

from networks.types import Params, InfoDict
from typing import Optional, Sequence, Tuple

import orbax.checkpoint
from flax.training import orbax_utils


def get_weight_decay_mask(params):
    flattened_params = flax.traverse_util.flatten_dict(
        flax.core.frozen_dict.unfreeze(params))

    def decay(k, v):
        if any([(key == 'bias' or 'Input' in key or 'Output' in key)
                for key in k]):
            return False
        else:
            return True

    return flax.traverse_util.unflatten_dict(
            {k: decay(k, v)
             for k, v in flattened_params.items()})


def get_weight_decay_mask_excludes(exclusions):
    """ Return a weight decay mask function that computes the pytree masks
        according to the given exclusion rules.
    """

    def decay(name, _):
        for rule in exclusions:
            if re.search(rule, jax.tree_util.keystr(name)) is not None:
                return False
        return True

    def weight_decay_mask(params):
        return jax.tree_util.tree_map_with_path(decay, params)

    return weight_decay_mask


@struct.dataclass
class Model(struct.PyTreeNode, ABC):
    step: int
    network: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    optimizer: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None
    """
    its network defines the computation tree for the inputs, it's a skeleton
    its params is an independent object
    its optimizer is defined from 'import optax | optax.adam'
    its forward/backward functions are conducted using the customized apply_gradient function
    use self.replace function modifies the  attributes
    
    step: record the number of gradient updates
    
    Model is initialized with Model.create() method
    """

    @classmethod
    def create(cls,
               network: nn.Module,
               inputs: Sequence[jnp.ndarray],  # sample of inputs
               optimizer: Optional[optax.GradientTransformation] = None,
               clip_grad_norm: float = None) -> 'Model':
        params = network.init(*inputs)  # (rng, other_inputs), params = {"params": ...}

        if optimizer is not None:
            if clip_grad_norm:
                optimizer = optax.chain(
                    optax.clip_by_global_norm(max_norm=clip_grad_norm),
                    optimizer)
            opt_state = optimizer.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   network=network,
                   params=params,
                   optimizer=optimizer,
                   opt_state=opt_state
                   )

    def __call__(self, *args, **kwargs):
        # the network defined by the jax nn.Model should be used by apply function with {'params': P} and other ..
        return self.network.apply(self.params, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.network.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple['Model', InfoDict]:
        grad_fn = jax.grad(loss_fn, has_aux=True)  # here auxiliary data is just the info dict
        grads, info = grad_fn(self.params)

        updates, new_opt_state = self.optimizer.update(grads, self.opt_state,
                                                       self.params)

        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info  # gives new model, info

    def get_state(self):
        return {"step": self.step}

    def save(self, save_path: str, force=True):
        os.makedirs(os.path.dirname(save_path), exist_ok=force)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        # need to set a new_model = model.load(...)
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)

    # def save(self, save_path: str, save_name: str = None):
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     save_args = orbax_utils.save_args_from_target(self)
    #     orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer(checkpoint_name=save_name)
    #     orbax_checkpointer.save(save_path, item=self, save_args=save_args, force=True)
    #
    # def load(self, load_path: str, save_name: str = None) -> 'Model':
    #     """
    #     This load require the correct definition of model structure.
    #     """
    #     orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer(checkpoint_name=save_name)
    #     return orbax_checkpointer.restore(load_path, item=self)


# if __name__ == '__main__':
#     from networks.mlp import MLP
#
#     # from flax.training import checkpoints
#     # from flax.training import orbax_utils
#
#     net_ = MLP((256, 256))
#     optx_ = optax.adamw(learning_rate=1e-4, mask=get_weight_decay_mask)
#     obs = jnp.zeros((32, 12))
#     rng = jax.random.PRNGKey(1)
#
#     vars_ = net_.init(rng, obs)
#     optx_.init(vars_)
#
#     actor = Model.create(net_, (rng, obs), optx_)
#
#     # single save & load
#     # it must provide the correct structure
#
#     orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
#     save_args = orbax_utils.save_args_from_target({"actor": actor, "data": jnp.ones((5,))})
#     orbax_checkpointer.save('ckpt/test', item={"actor": actor, "data": jnp.ones((5,))}, save_args=save_args, force=True)
#     target = {"actor": actor, "data": jnp.zeros((5,))}
#     raw_restored = orbax_checkpointer.restore('ckpt/test', item=target)
#     raw_restored
#     Model.create(raw_restored)
#
#     # training save
#     options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
#     checkpoint_manager = orbax.checkpoint.CheckpointManager(
#         '/tmp/flax_ckpt/orbax/managed', orbax_checkpointer, options)
#
#     save_args = orbax_utils.save_args_from_target(actor)
#     orbax_checkpointer.save('ckpt/test', actor, save_args=save_args, checkpoint_name='hi')
