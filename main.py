import os
import ray
import random
import numpy as np
from collections import deque
from plots import plot_curve
import argparse

parser = argparse.ArgumentParser(description='Offline reinforcement learning')
parser.add_argument('--gpu', default='6, 7', type=str, help='The device to use.')
# 'walker2d-expert-v2'  'halfcheetah-expert-v2' 'ant-medium-v2'    hopper-medium-v2
parser.add_argument('--env', default='hopper-medium-replay-v2', type=str, help='Environment name.')
parser.add_argument('--reward_tune', default='iql_locomotion', type=str, help='Reward tune.')
parser.add_argument('--dataset_name', default='d4rl', choices=['d4rl'], help='Dataset name.')
parser.add_argument('--agent', default='dac', type=str, help='Training methods')
parser.add_argument('--seed', default=0, type=int, help='Random seed.')
parser.add_argument('--num_seeds', default=5, type=int, help='number of runs for different seeds')
parser.add_argument('--n_eval_episodes', default=10, type=int, help='Number of episodes used for evaluation.')
parser.add_argument('--log_interval', default=5000, type=int, help='Logging interval.')
parser.add_argument('--eval_interval', default=10000, type=int, help='Eval interval.')
parser.add_argument('--batch_size', default=256, type=int, help='Mini batch size.')
parser.add_argument('--max_steps', default=int(2e6), type=int, help='Number of training steps.')
parser.add_argument('--finetune_step', default=int(3e6), type=int,
                    help='After which it will change to online fine-tuning')
parser.add_argument('--buffer_size', default=int(1e6), type=int, help='The replay buffer size of online fine-tuning')
parser.add_argument('--discount', default=0.99, type=float, help='Discount factor')
parser.add_argument('--percentile', default=100.0, type=float,
                    help='Dataset percentile (see https://arxiv.org/abs/2106.01345).')
parser.add_argument('--percentage', default=100.0, type=float, help='Percentage of the dataset to use for training.')
parser.add_argument('--no_tqdm', action="store_false", help='Disable tqdm progress bar.')
parser.add_argument('--save_video', action="store_true", help='Save videos during evaluation.')
parser.add_argument('--save_ckpt', action="store_true", help='Save agents during training.')
parser.add_argument('--test', action="store_true", help='Activate test mode. without ray process')
parser.add_argument('--rand_batch', action="store_true", help='Scanning or random batch sampling of the dataset')
parser.add_argument('--temperature', default=0, type=float,
                    help='Use argmax (=0) or random action according to temperature')
parser.add_argument('--tag', default='', type=str, help='Give a tag to name specific experiment.')

# model configs
parser.add_argument('--T', default=5, type=int, help='The total number of diffusion steps.')
parser.add_argument('--eta', default=1, type=float, help='Weights of BC term. It also defines the initial eta value')
parser.add_argument('--eta_min', default=0.001, type=float, help='The minimal value of eta')
parser.add_argument('--eta_max', default=100., type=float, help='The maximal value of eta')
parser.add_argument('--eta_lr', default=0., type=float, help='The learning rate of dual gradient ascent for eta')
parser.add_argument('--rho', default=1, type=float, help='The weight of lower confidence bound.')
parser.add_argument('--bc_threshold', default=1, type=float, help='threshold to control eta for bc loss')
parser.add_argument('--actor_lr', default=3e-4, type=float, help='learning rate for actor network')
parser.add_argument('--critic_lr', default=3e-4, type=float, help='learning rate for critic network')
parser.add_argument('--ema_tau', default=0.005, type=float, help='learning rate for exponential moving average.')
parser.add_argument('--q_tar', default='lcb', type=str, help='The type of Q target')
parser.add_argument('--Q_guidance', default='soft', choices=['soft', 'hard', 'denoised'],
                    help='Types of Q-gradient guidance.')
parser.add_argument('--maxQ', action="store_true", help='Whether taking max Q over actions during critic learning')
parser.add_argument('--resnet', action="store_true", help='Whether to use MLPResNet as noise models')
parser.add_argument('--num_qs', default=10, type=int, help='The number of Q heads')
parser.add_argument('--num_q_samples', default=10, type=int,
                    help='The number of actions samples for Q-target estimation')
parser.add_argument('--num_action_samples', default=10, type=int, help='The number of Q samples')
FLAGS = parser.parse_args()

# set computing resources
NUM_GPUS = len(str(FLAGS.gpu).split(','))
SEEDS_PER_GPU = FLAGS.num_seeds // NUM_GPUS + int(FLAGS.num_seeds % NUM_GPUS != 0)
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(round(1 / SEEDS_PER_GPU, 2))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
ray.init(log_to_driver=False)  # ignore some warnings (DeprecationWarning) to make it clean

from eval import eval_agent, STATISTICS
from utils import make_env
from utils import prepare_output_dir, MBars
from tensorboardX import SummaryWriter  # after ray, otherwise it will limit the cpu usage
from datasets import make_env_and_dataset, ReplayBuffer  # after ray, otherwise no gpu is used


def _seed_run(learner,
              config: dict,
              # dataset,  # shared memory: do not change its attributes in the seed_run!
              save_dir,
              pbar=None,
              idx=0,
              buffer_size=int(1e6)):
    local_seed = config['seed'] * 100 + idx

    random.seed(local_seed)
    np.random.seed(local_seed)

    with open(os.path.join(save_dir, f"seed_{local_seed}.txt"), "w") as f:
        print("\t".join(["steps"] + STATISTICS), file=f)

    summary_writer = SummaryWriter(os.path.join(save_dir, 'tensorboard', 'seed_' + str(local_seed)))
    video_save_folder = None if not config['save_video'] else os.path.join(
        save_dir, 'video', 'eval')
    ckpt_save_folder = os.path.join(save_dir, 'ckpt')

    # for evaluation
    env = make_env(config['env'], local_seed, video_save_folder)

    # separate dataset for each process
    _, dataset, reward_fn = make_env_and_dataset(config['env'], config['seed'], config['dataset_name'],
                                                 reward_tune=config["reward_tune"],
                                                 scanning=not config['rand_batch'])

    if config['percentage'] < 100.0:
        dataset.take_random(config['percentage'])

    if config['percentile'] < 100.0:
        dataset.take_top(config['percentile'])

    # for online fine-tuning, fix the size of the replay buffer to be 1 million
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space, capacity=buffer_size)
    replay_buffer.initialize_with_dataset(dataset, num_samples=buffer_size)
    replay_buffer.reward_fn = reward_fn
    finetune_env = make_env(config['env'], local_seed + 100, video_save_folder)
    observation, done = finetune_env.reset(), False

    a_config = config.copy()
    a_config['seed'] = local_seed
    agent = learner(env.observation_space.sample()[np.newaxis],  # given a batch dim, shape = [1, *(raw_shape)]
                    env.action_space.sample()[np.newaxis],
                    lr_decay_steps=config['max_steps'],
                    **a_config)

    last_window_mean_return = deque(maxlen=5)
    try:
        running_max_return = float("-inf")
        for i in range(config['max_steps']):

            if i % config['eval_interval'] == 0:
                eval_res = eval_agent(i, agent, env, summary_writer, save_dir, local_seed,
                                      config['n_eval_episodes'])
                last_window_mean_return.append(eval_res['mean'])
                if eval_res['mean'] > running_max_return:
                    running_max_return = eval_res['mean']
                    if config['save_ckpt']:
                        agent.save_ckpt(prefix=f'{idx}_best_', ckpt_folder=ckpt_save_folder, silence=True)
                if config['save_ckpt']:
                    agent.save_ckpt(prefix=f'{idx}_eval_', ckpt_folder=ckpt_save_folder, silence=True)

            if i < config['finetune_step']:
                # offline RL
                batch = dataset.sample(batch_size=config['batch_size'])
            else:
                # online fine-tuning
                action = agent.sample_actions(observation, temperature=0)
                next_observation, reward, done, info = finetune_env.step(np.clip(action, -1, 1))
                if not done or 'TimeLimit.truncated' in info:
                    mask = 1.0
                else:
                    mask = 0.0
                replay_buffer.insert(observation, action, reward, mask,
                                     float(done), next_observation)
                observation = next_observation
                if done:
                    observation, done = finetune_env.reset(), False
                batch = replay_buffer.sample(batch_size=config['batch_size'])

            update_info = agent.update(batch)
            if i % config['log_interval'] == 0:  #
                for k, v in update_info.items():
                    if v.ndim == 0:
                        summary_writer.add_scalar(f'training/{k}', v, i)
                    else:
                        summary_writer.add_histogram(f'training/{k}', v, i)
                summary_writer.flush()

            if pbar is not None:
                pbar.update.remote(idx)

        # return final evaluations
        final_eval = eval_agent(config['max_steps'], agent, env, summary_writer, save_dir, local_seed,
                                config['n_eval_episodes'])

        last_window_mean_return.append(final_eval['mean'])

        if config['save_ckpt']:  # save final checkpoints
            agent.save_ckpt(prefix=f'{idx}finished_', ckpt_folder=ckpt_save_folder, silence=True)

        return np.mean(last_window_mean_return), running_max_return

    except (KeyboardInterrupt, RuntimeError) as m_:
        # save checkpoints if interrupted
        print("Stopped by exception:", m_)
        if config['save_ckpt']:
            agent.save_ckpt(prefix=f'{idx}_expt_', ckpt_folder=ckpt_save_folder, silence=True)


@ray.remote(num_gpus=1 / SEEDS_PER_GPU, num_cpus=1)
def seed_run(*args, **kwargs):
    return _seed_run(*args, **kwargs)


def main():
    # config = __import__(f'configs.{FLAGS.agent}_config', fromlist=('configs', FLAGS.agent)).config
    config = vars(FLAGS)
    if FLAGS.eta_lr > 0:
        tag = f"BC<={FLAGS.bc_threshold}|QTar={FLAGS.q_tar}|rho={FLAGS.rho}|{FLAGS.tag}" if str(
            FLAGS.agent) == 'dac' else str(
            FLAGS.tag)
    else:
        tag = f"eta={FLAGS.eta}|QTar={FLAGS.q_tar}|rho={FLAGS.rho}|{FLAGS.tag}" if str(FLAGS.agent) == 'dac' else str(
            FLAGS.tag)
    save_dir = prepare_output_dir(folder=os.path.join('results', FLAGS.env),
                                  time_stamp=True,
                                  suffix=str(FLAGS.agent).upper() + tag)

    print('=' * 10 + ' Arguments ' + '=' * 10)
    with open(os.path.join(save_dir, 'config.txt'), 'w') as file:
        for k, v in config.items():
            value = str(v)
            print(k + ' = ' + value)
            print(k + ' = ' + value, file=file)
        print(f"Save_folder = {save_dir}", file=file)
    print(f"\nSave results to: {save_dir}\n")

    # _, dataset, r_fn = make_env_and_dataset(FLAGS.env, FLAGS.seed, FLAGS.dataset_name,
    #                                         reward_tune=config["reward_tune"],
    #                                         scanning=not FLAGS.rand_batch)
    #
    # if config['percentage'] < 100.0:
    #     dataset.take_random(config['percentage'])
    #
    # if config['percentile'] < 100.0:
    #     dataset.take_top(config['percentile'])

    import agents
    learner = {'bc': agents.BCLearner,
               'iql': agents.IQLLearner,
               'sac': agents.SACLearner,
               'ivr': agents.IVRLearner,
               'dbc': agents.DDPMBCLearner,
               'dac': agents.DACLearner,
               'dql': agents.DQLLearner,
               'cql': agents.CQLLearner}[FLAGS.agent]

    if FLAGS.test:
        print("start testing!")
        # flag_dict = FLAGS.flag_values_dict()
        config['max_steps'] = 10
        config['eval_interval'] = 5
        config['seed'] = 123
        _seed_run(learner,
                  config,
                  save_dir,
                  None,
                  0,
                  FLAGS.buffer_size)
        print("testing passed!")
        return
    pbar = MBars(FLAGS.max_steps, ':'.join([FLAGS.env, FLAGS.agent, tag]), FLAGS.num_seeds)

    futures = [seed_run.remote(learner,
                               config,
                               save_dir,
                               pbar.process,
                               i,
                               FLAGS.buffer_size)  # send dict to the multiprocessing
               for i in range(FLAGS.num_seeds)]
    pbar.flush()
    final_res = ray.get(futures)
    final_scores, running_best = [_[0] for _ in final_res], [_[1] for _ in final_res]
    # statistics over different seeds
    f_mean, f_std, f_max, f_min = np.mean(final_scores), np.std(final_scores), max(final_scores), min(final_scores)
    b_mean, b_std, b_max, b_min = np.mean(running_best), np.std(running_best), max(running_best), min(running_best)
    print(f'Final eval: mean={np.mean(final_scores)}, std={np.std(final_scores)}')

    # record the final results of different seeds
    with open(os.path.join(save_dir, f"final_mean_scores.txt"), "w") as f:
        print("\t".join([f"seed_{FLAGS.seed + _}" for _ in range(FLAGS.num_seeds)] + ["mean", "std", "max", "min"]),
              file=f)
        print("\t".join([str(round(_, 2)) for _ in final_scores] + [str(f_mean), str(f_std), str(f_max), str(f_min)]),
              file=f)
        print("\t".join([str(round(_, 2)) for _ in running_best] + [str(b_mean), str(b_std), str(b_max), str(b_min)]),
              file=f)

    fig, ax = plot_curve(save_dir, label=":".join([FLAGS.agent, FLAGS.env]))
    fig.savefig(os.path.join(save_dir, "training_curve.png"))


if __name__ == '__main__':
    main()
