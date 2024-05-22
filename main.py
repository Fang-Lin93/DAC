import os
import ray

os.environ["CUDA_VISIBLE_DEVICES"] = "7, 6"
# init ray here to ensure the visible gpus and the proper loading of jax
ray.init(log_to_driver=False)  # ignore some warnings (DeprecationWarning) to make it clean

import random
from tensorboardX import SummaryWriter
import numpy as np
from agents import *
from datasets import make_env_and_dataset, ReplayBuffer
from utils import make_env
from utils import prepare_output_dir, MBars
from eval import eval_agent, STATISTICS
from plots import plot_curve
from collections import deque
from absl import app, flags


FLAGS = flags.FLAGS
# 'walker2d-expert-v2'  'halfcheetah-expert-v2' 'ant-medium-v2'    hopper-medium-v2
flags.DEFINE_string('env', 'hopper-medium-replay-v2', 'Environment name.')
flags.DEFINE_string('reward_tune', 'iql_locomotion', 'Reward tune.')
flags.DEFINE_enum('dataset_name', 'd4rl', ['d4rl'], 'Dataset name.')
flags.DEFINE_enum('agent', 'dac',
                  ['bc', 'iql', 'sac', 'ivr', 'hql', 'dbc', 'qcd', 'dql', 'dac'], 'Training methods')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_seed_runs', 8, 'number of runs for different seeds')
flags.DEFINE_integer('n_eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(2e6), 'Number of training steps.')
flags.DEFINE_integer('finetune_step', int(3e6), 'When to change to online fine-tuning (future work)')
flags.DEFINE_integer('buffer_size', int(1e6), 'The replay buffer size of online fine-tuning')
flags.DEFINE_float('discount', 0.99, 'Discount factor')
flags.DEFINE_float('percentile', 100.0, 'Dataset percentile (see https://arxiv.org/abs/2106.01345).')
flags.DEFINE_float('percentage', 100.0, 'Percentage of the dataset to use for training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('save_ckpt', False, 'Save agents during training.')
flags.DEFINE_boolean('test', False, 'activate test mode. without ray process')
flags.DEFINE_boolean('rand_batch', False, 'Scanning or random batch sampling of the dataset')
flags.DEFINE_float('temperature', 0, 'Use argmax (=0) or random action according to temperature')
flags.DEFINE_string('tag', '', 'Give a tag to name specific experiment.')


# model configs
flags.DEFINE_integer('T', 5, 'The total number of diffusion steps.')
flags.DEFINE_float('eta', 1, 'Weights of BC term. It also defines the initial eta value')
flags.DEFINE_float('eta_min', 0.001, 'The minimal value of eta')
flags.DEFINE_float('eta_max', 100., 'The maximal value of eta')
flags.DEFINE_float('eta_lr', 0., 'The learning rate of dual gradient ascent for eta')
flags.DEFINE_float('rho', 1, 'The weight of lower confidence bound.')
flags.DEFINE_float('bc_threshold', 1, 'threshold to control eta for bc loss')
flags.DEFINE_float('actor_lr', 3e-4, 'learning rate for actor network')
flags.DEFINE_float('critic_lr', 3e-4, 'learning rate for critic network')
flags.DEFINE_float('ema_tau', 0.005, 'learning rate for exponential moving average.')
flags.DEFINE_string('q_tar', 'lcb', 'The type of Q target')
flags.DEFINE_enum('Q_guidance', 'soft', ['soft', 'hard', 'denoised'], 'Types of Q-gradient guidance.')
flags.DEFINE_boolean('maxQ', False, 'Whether taking max Q over actions during critic learning')
flags.DEFINE_boolean('resnet', False, 'Whether to use MLPResNet as noise models')
flags.DEFINE_integer('num_qs', 10, 'The number of Q heads')
flags.DEFINE_integer('num_q_samples', 10, 'The number of actions samples for Q-target estimation')
flags.DEFINE_integer('num_action_samples', 10, 'The number of Q samples')

Learner = {'bc': BCLearner,
           'iql': IQLLearner,
           'sac': SACLearner,
           'ivr': IVRLearner,
           'dbc': DDPMBCLearner,
           'dac': DACLearner,
           'dql': DQLLearner}


def _seed_run(learner,
              config: dict,
              dataset,
              save_dir,
              pbar=None,
              idx=0,
              reward_fn=None,
              buffer_size=int(1e6)):
    # record eval stats
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

    # for online fine-tuning, fix the size of the replay buffer to be 1 million
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space, capacity=buffer_size)
    replay_buffer.initialize_with_dataset(dataset, num_samples=buffer_size)
    replay_buffer.reward_fn = reward_fn
    finetune_env = make_env(config['env'], local_seed + 100, video_save_folder)
    observation, done = finetune_env.reset(), False

    if config['percentage'] < 100.0:
        dataset.take_random(config['percentage'])

    if config['percentile'] < 100.0:
        dataset.take_top(config['percentile'])

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
                batch = dataset.sample(config['batch_size'])
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
                batch = replay_buffer.sample(config['batch_size'])

            update_info = agent.update(batch)
            if i % config['log_interval'] == 0:
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

        if config['save_ckpt']:    # save final checkpoints
            agent.save_ckpt(prefix=f'{idx}finished_', ckpt_folder=ckpt_save_folder, silence=True)

        return np.mean(last_window_mean_return), running_max_return

    except (KeyboardInterrupt, RuntimeError) as m_:
        # save checkpoints if interrupted
        print("Stopped by exception:", m_)
        if config['save_ckpt']:
            agent.save_ckpt(prefix=f'{idx}_expt_', ckpt_folder=ckpt_save_folder, silence=True)


@ray.remote(num_gpus=0.24)
def seed_run(*args, **kwargs):
    return _seed_run(*args, **kwargs)


def main(_):
    config = __import__(f'configs.{FLAGS.agent}_config', fromlist=('configs', FLAGS.agent)).config
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
    # update config if is specified in FLAG -> save config
    config.update(FLAGS.flag_values_dict())
    print('=' * 10 + ' Arguments ' + '=' * 10)
    with open(os.path.join(save_dir, 'config.txt'), 'w') as file:
        for k, v in config.items():
            # if hasattr(FLAGS, k):
            #     value = str(getattr(FLAGS, k)) + "*"  # re-claimed in FLAG definition
            # else:
            value = str(v)
            print(k + ' = ' + value)
            print(k + ' = ' + value, file=file)
        print(f"Save_folder = {save_dir}", file=file)
    print(f"\nSave results to: {save_dir}\n")

    _, dataset, r_fn = make_env_and_dataset(FLAGS.env, FLAGS.seed, FLAGS.dataset_name,
                                            reward_tune=config["reward_tune"],
                                            scanning=not FLAGS.rand_batch)
    learner = Learner[FLAGS.agent]

    if FLAGS.test:
        print("start testing!")
        # flag_dict = FLAGS.flag_values_dict()
        config['max_steps'] = 10
        config['eval_interval'] = 5
        config['seed'] = 123
        _seed_run(learner,
                  config,
                  dataset,
                  save_dir,
                  None,
                  0,
                  r_fn,
                  FLAGS.buffer_size)
        print("testing passed!")
        return
    pbar = MBars(FLAGS.max_steps, ':'.join([FLAGS.env, FLAGS.agent, tag]), FLAGS.num_seed_runs)

    futures = [seed_run.remote(learner,
                               config,
                               dataset,
                               save_dir,
                               pbar.process,
                               i,
                               r_fn)  # send dict to the multiprocessing
               for i in range(FLAGS.num_seed_runs)]
    pbar.flush()
    final_res = ray.get(futures)
    final_scores, running_best = [_[0] for _ in final_res], [_[1] for _ in final_res]
    # statistics over different seeds
    f_mean, f_std, f_max, f_min = np.mean(final_scores), np.std(final_scores), max(final_scores), min(final_scores)
    b_mean, b_std, b_max, b_min = np.mean(running_best), np.std(running_best), max(running_best), min(running_best)
    print(f'Final eval: mean={np.mean(final_scores)}, std={np.std(final_scores)}')

    # record the final results of different seeds
    with open(os.path.join(save_dir, f"final_mean_scores.txt"), "w") as f:
        print("\t".join([f"seed_{FLAGS.seed + _}" for _ in range(FLAGS.num_seed_runs)] + ["mean", "std", "max", "min"]),
              file=f)
        print("\t".join([str(round(_, 2)) for _ in final_scores] + [str(f_mean), str(f_std), str(f_max), str(f_min)]),
              file=f)
        print("\t".join([str(round(_, 2)) for _ in running_best] + [str(b_mean), str(b_std), str(b_max), str(b_min)]),
              file=f)

    fig, ax = plot_curve(save_dir, label=":".join([FLAGS.agent, FLAGS.env]))
    fig.savefig(os.path.join(save_dir, "training_curve.png"))


if __name__ == '__main__':
    app.run(main)
