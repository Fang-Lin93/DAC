import numpy as np
import gym
import os
from typing import Dict

STATISTICS = ["mean", "median", "std", "min", "max"]


def evaluate(agent, env: gym.Env, num_episodes: int, raw_results: bool = False) -> Dict:
    returns, lengths, info = [], [], {}
    for _ in range(num_episodes):
        observation, done = env.reset(), False

        while not done:
            action = agent.sample_actions(observation, temperature=0)  # eval takes argmax from actor net
            observation, reward, done, info = env.step(np.clip(action, -1, 1))

        # episodic statistics from wrappers/EpisodeMonitor
        returns.append(info['episode']['return'])
        lengths.append(info['episode']['length'])

    if raw_results:
        return {'returns': returns,
                'lengths': lengths}

    return {
        'mean': np.mean(returns),
        'median': np.median(returns),
        'std': np.std(returns),
        'min': np.min(returns),
        'max': np.max(returns),
        'length': np.mean(lengths)
    }


def eval_agent(step, agent, env, summary_writer, save_dir, seed, eval_episodes):
    eval_stats = evaluate(agent, env, eval_episodes)

    for k, v in eval_stats.items():
        summary_writer.add_scalar(f'evaluation/average_{k}s', v, step)
    summary_writer.flush()

    with open(os.path.join(save_dir, f"seed_{seed}.txt"), "a+") as f:
        stats = [step] + [eval_stats[_] for _ in STATISTICS]
        print("\t".join([str(round(_, 2)) for _ in stats]), file=f)
    return eval_stats
