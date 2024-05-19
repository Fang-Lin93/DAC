

import os
import jax.numpy as jnp
import numpy as np
from datasets import Batch
from networks.types import InfoDict
from flax.training import orbax_utils
import orbax


"""
Base agent
"""


class Agent(object):
    name = 'agent'
    model_names = []

    def update(self, batch: Batch) -> InfoDict:
        """
        self.rng, self.bc_diffuser, info = jit_update_diffusion_model(self.bc_diffuser, batch, self.rng, self.T,
                                                                      self.alpha_hats)
        self._n_training_steps += 1
        :param batch:
        :return: info
        """
        raise NotImplementedError

    def sample_actions(self,
                       observations: jnp.ndarray,
                       temperature: float = 0) -> jnp.ndarray:

        raise NotImplementedError

    def save_ckpt(self, prefix: str, ckpt_folder: str = "ckpt", silence: bool = True,
                  legacy: bool = False):
        """
        load & save models by the model attribute names
        """
        assert prefix
        # save_dir = os.path.join(ckpt_folder, self.name, prefix)
        save_dir = os.path.join(os.getcwd(), ckpt_folder, self.name)
        for n_ in self.model_names:
            save_target = os.path.join(save_dir, prefix + n_)
            if not legacy:
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                ckpt = {'model': self.__getattribute__(n_)}
                save_args = orbax_utils.save_args_from_target(ckpt)
                orbax_checkpointer.save(save_target, ckpt, save_args=save_args, force=True)

            else:
                self.__getattribute__(n_).save(save_target)

            if not silence:
                print(f"Successfully save {n_} model to {save_target}")

        # for n_ in self.model_names:
        #     self.__getattribute__(n_).save(os.path.join(save_dir, prefix + n_))
        #     if not silence:
        #         print(f"Successfully save {n_} model to {os.path.join(save_dir, prefix + n_)}")

    def load_ckpt(self, prefix: str = "", ckpt_folder: str = "ckpt", silence: bool = False,
                  legacy: bool = False):
        # save_dir = os.path.join(ckpt_path, self.name, prefix)
        assert prefix
        save_dir = os.path.join(os.getcwd(), ckpt_folder, self.name)
        for n_ in self.model_names:
            restore_target = os.path.join(save_dir, prefix + n_)
            if not legacy:
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                target = {'model': self.__getattribute__(n_)}  # use raw as the tree prototype
                restore_model = orbax_checkpointer.restore(restore_target, item=target)
                self.__setattr__(n_, restore_model['model'])
            else:
                self.__setattr__(n_, self.__getattribute__(n_).load(restore_target))
            if not silence:
                print(f"Successfully load {n_} model from {restore_target}")

        # for n_ in self.model_names:
        #     model = self.__getattribute__(n_)
        #     self.__setattr__(n_, model.load(os.path.join(save_dir, prefix + n_)))
        #     if not silence:
        #         print(f"Successfully load {n_} model from {os.path.join(save_dir, prefix + n_)}")
