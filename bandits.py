import os
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from utils import sample_n_k
from tqdm import trange
from agents import DACLearner, DDPMBCLearner
from datasets import Batch
from sklearn import datasets

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

N_epochs = 20000
lr = 1e-3
batch_size = 128
T = 50
bc_threshold = 1.3


class Dataset(object):
    # action_dim = 2

    def __init__(self,
                 seed: int,
                 idx: int = 0):
        self.samples_per_center = 400
        self.idx = idx

        print(f"Use bandit dataset={idx}")
        np.random.seed(seed)

        if idx == 0:
            self.centers = [
                jnp.array((-0.6, 0.6)),
                jnp.array((0.4, 0.2)),
            ]
            self.scale = 0.1

        elif idx == 1:
            self.centers = [
                jnp.array((-0.6, 0.6)),
                jnp.array((0.4, 0.4)),
                jnp.array((-0.4, -0.4)),
            ]
            self.scale = 0.1

        elif idx == 2:
            centers, _ = datasets.make_moons(n_samples=40)
            centers = (centers - centers.min(axis=0)) / (centers.max(axis=0) - centers.min(axis=0)) * 1.1
            self.centers = centers + jnp.array((-0.8, -0.2))
            self.samples_per_center = 10
            self.scale = 0.05

        elif idx == 3:
            centers, _ = datasets.make_circles(n_samples=40)
            centers = (centers - centers.min(axis=0)) / (centers.max(axis=0) - centers.min(axis=0)) * 1.1
            self.centers = centers + jnp.array((-0.8, -0.2))
            self.samples_per_center = 10
            self.scale = 0.05

        rng = jax.random.PRNGKey(seed)
        self.rng, *keys = jax.random.split(rng, 1 + len(self.centers))
        mods = [jax.random.normal(k_, (self.samples_per_center, 2)) for k_ in keys]

        self.actions = jnp.concatenate([c_ + self.scale * m_ for c_, m_ in zip(self.centers, mods)], axis=0).clip(-1, 1)
        self.data_size = len(self.actions)

        # Q-surfaces
        dist_fn = lambda a0, a1, cx, cy: - jnp.sqrt((a0 - cx) ** 2 + (a1 - cy) ** 2)  # * 10
        centers = [(0.4, -0.4)]
        self.Q_surface = jax.vmap(lambda a0, a1: sum(dist_fn(a0, a1, cx, cy) for (cx, cy) in centers))
        # self.Q_surface = jax.vmap(lambda a0, a1: a0 - a1)

    def sample(self, batch_size: int) -> Batch:
        sampled_ids = sample_n_k(self.data_size, batch_size)
        sampled_actions = self.actions[sampled_ids]
        self.rng, key = jax.random.split(self.rng)
        reward_noise = jax.random.normal(key, (batch_size,)) * 0.5
        return Batch(observations=jnp.zeros((batch_size, 1)),
                     actions=sampled_actions,
                     rewards=self.Q_surface(sampled_actions[:, 0], sampled_actions[:, 1]) + reward_noise,
                     masks=jnp.zeros((batch_size,)),
                     next_observations=jnp.zeros((batch_size, 1)))

    def plot_data_points(self):
        plt.scatter(self.actions[:, 0], self.actions[:, 1], s=3,
                    c=self.Q_surface(self.actions[:, 0], self.actions[:, 1]))
        plt.colorbar()
        # c=self.Q_surface(*jnp.meshgrid(self.actions)), cmap='cool'
        # plot level curve
        nxy = 100
        x, y = jnp.meshgrid(jnp.linspace(-1, 1, nxy), jnp.linspace(-1, 1, nxy))
        z = self.Q_surface(x, y)
        plt.contour(x, y, z)
        plt.show()


def reset_models():
    dbc = DDPMBCLearner(seed=520,
                        hidden_dims=(64, 64, 64),
                        observations=jnp.zeros((1, 1)),
                        actions=jnp.zeros((1, 2)),
                        sampler='ddpm',
                        lr_decay_steps=N_epochs,
                        actor_lr=lr,
                        T=T
                        )

    hard = DACLearner(seed=520,
                      hidden_dims=(64, 64, 64),
                      observations=jnp.zeros((1, 1)),
                      actions=jnp.zeros((1, 2)),
                      sampler='ddpm',
                      eta=1,
                      eta_lr=0.01,
                      bc_threshold=bc_threshold,
                      q_tar='lcb',
                      Q_guidance="hard",
                      use_guidance_loss=True,
                      num_q_samples=10,
                      num_action_samples=1,
                      actor_lr=lr,
                      critic_lr=lr,
                      step_start_ema=10,
                      T=T,
                      temperature=0,
                      act_with_q_guid=False,
                      clip_sampler=False,
                      num_qs=5,
                      )

    denoised = DACLearner(seed=520,
                          hidden_dims=(64, 64, 64),
                          observations=jnp.zeros((1, 1)),
                          actions=jnp.zeros((1, 2)),
                          sampler='ddpm',
                          eta=1,
                          eta_lr=0.01,
                          bc_threshold=bc_threshold,
                          q_tar='lcb',
                          Q_guidance="denoised",
                          use_guidance_loss=True,
                          num_q_samples=10,
                          num_action_samples=1,
                          actor_lr=lr,
                          critic_lr=lr,
                          step_start_ema=10,
                          T=T,
                          temperature=0,
                          act_with_q_guid=False,
                          clip_sampler=False,
                          num_qs=5,
                          )

    soft = DACLearner(seed=520,
                      hidden_dims=(64, 64, 64),
                      observations=jnp.zeros((1, 1)),
                      actions=jnp.zeros((1, 2)),
                      sampler='ddpm',
                      eta=1,
                      eta_lr=0.01,
                      bc_threshold=bc_threshold,
                      q_tar='lcb',
                      Q_guidance="soft",
                      use_guidance_loss=True,
                      num_q_samples=10,
                      num_action_samples=1,
                      actor_lr=lr,
                      critic_lr=lr,
                      step_start_ema=10,
                      T=T,
                      temperature=0,
                      act_with_q_guid=False,
                      clip_sampler=False,
                      num_qs=5,
                      )

    return dbc, hard, denoised, soft


if __name__ == '__main__':

    new_data = False
    if not os.path.exists('results/bandits/data'):
        os.makedirs('results/bandits/data')

    print(f"new_data={new_data}")

    fig, ax = plt.subplots(1, 4, figsize=(25, 6))
    fig.tight_layout()
    fig.subplots_adjust(right=1.11, wspace=0.1)
    for i in range(4):
        obs = jnp.zeros((100, 1))  # why 50 not worked?
        # plots
        nxy = 100
        x, y = jnp.meshgrid(jnp.linspace(-1.5, 1.5, nxy), jnp.linspace(-1.5, 1.5, nxy))
        xy = jnp.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1))], axis=-1)
        data = Dataset(seed=0, idx=i)
        true_z = data.Q_surface(x, y)

        DBC, Hard, Denoised, Soft = reset_models()

        if new_data:
            for _ in trange(N_epochs):
                DBC.update(data.sample(batch_size))
                Hard.update(data.sample(batch_size))
                Denoised.update(data.sample(batch_size))
                Soft.update(data.sample(batch_size))

            pred0 = DBC.sample_actions(observations=obs, batch_act=True).clip(-1.45, 1.45)
            pred1 = Denoised.sample_actions(observations=obs, batch_act=True).clip(-1.45, 1.45)
            pred2 = Hard.sample_actions(observations=obs, batch_act=True).clip(-1.45, 1.45)
            pred3 = Soft.sample_actions(observations=obs, batch_act=True).clip(-1.45, 1.45)
            # predict value surface
            pred_z = Soft.critic(jnp.zeros((nxy ** 2, 1)), xy).mean(axis=0).reshape((nxy, nxy))

            np.save(f'results/bandits/data/{i}_0.npy', pred0)
            np.save(f'results/bandits/data/{i}_1.npy', pred1)
            np.save(f'results/bandits/data/{i}_2.npy', pred2)
            np.save(f'results/bandits/data/{i}_3.npy', pred3)
            np.save(f'results/bandits/data/{i}_f.npy', pred_z)

        else:
            print("use trained data")
            pred0 = np.load(f'results/bandits/data/{i}_0.npy')
            pred1 = np.load(f'results/bandits/data/{i}_1.npy')
            pred2 = np.load(f'results/bandits/data/{i}_2.npy')
            pred3 = np.load(f'results/bandits/data/{i}_3.npy')
            pred_z = np.load(f'results/bandits/data/{i}_f.npy')

        ax[i].set_aspect('equal', 'box')
        # true_levels = ax[i].contourf(x, y, true_z)  # truth field
        levels = ax[i].contour(x, y, pred_z, linestyles='--', cmap="binary")  # cmap='Blues' cividis
        ax[i].clabel(levels, levels.levels, inline=True, fmt=lambda v_: f'{v_:.2f}', fontsize=16)

        scatter = ax[i].scatter(data.actions[:, 0], data.actions[:, 1],  # c='black', cmap='autumn',
                                     # c='black',
                                     c=data.Q_surface(data.actions[:, 0], data.actions[:, 1]),
                                     marker='.',
                                     label='Behavior Data',)

        # pred
        # ax[i].scatter(pred0[:, 0], pred0[:, 1], marker='x', c='orange', label='Behavior Cloning')
        ax[i].scatter(pred1[:, 0], pred1[:, 1], marker='x', c='orange', label=f'Denoised Guidance')
        ax[i].scatter(pred2[:, 0], pred2[:, 1], marker='x', c='blue', label='Hard Q-Guidance')
        ax[i].scatter(pred3[:, 0], pred3[:, 1], marker='^', c='magenta', label=f'Soft Q-Guidance')
        ax[i].set_xlim(-1.5, 1.5)
        ax[i].set_ylim(-1.5, 1.5)

        if i == 0:
            ax[i].legend(loc='lower left', fontsize=10)

        if i == 3:
            fig.colorbar(scatter, ax=ax, shrink=0.9, pad=0.01, aspect=30).set_label("Reward Value", fontsize=24)

    if not os.path.exists('results/bandits'):
        os.makedirs('results/bandits')
    fig.savefig(f'results/bandits/bandits.png')
    fig.savefig(f'results/bandits/bandits.svg')
    fig.show()

