import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os


def plot_curve(dirname,
               fig=None,
               ax=None,
               title=None,
               curve="mean",
               confidence_interval=True,
               label=None,
               window_size=1,
               ):
    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.tight_layout()

    if title is None:
        title = f"{curve} performance"

    f_list = [os.path.join(dirname, _) for _ in os.listdir(dirname) if _.endswith('txt') and 'seed' in _]

    records = pd.DataFrame()
    for f in f_list:
        rec_ = pd.DataFrame(pd.read_csv(f, sep='\t', index_col='steps'))[[curve]]
        records = pd.concat([records, rec_], axis=1)

    records = records.dropna(axis=0)

    x_axis = records.index[window_size - 1:].to_numpy()

    mean = records[curve].mean(axis=1).to_numpy() if len(f_list) > 1 else records[curve].to_numpy()

    if window_size > 1:
        mean = [np.mean(mean[i: i + window_size]) for i in range(len(mean) - window_size + 1)]
    ax.plot(x_axis, mean, label=label if label is not None else dirname.split('_')[-1], alpha=0.9)

    if confidence_interval and curve == "mean":
        std = records[curve].std(axis=1).to_numpy()[:len(mean)] if len(f_list) > 1 else 0
        upper, lower = mean + std, mean - std
        ax.fill_between(x_axis, lower, upper, alpha=0.1)

    ax.set_xlabel("steps")
    ax.set_ylabel(f"Evaluation {curve}")
    ax.set_title(title)
    ax.legend()
    return fig, ax


def compare_curves(dirs):
    fig_, ax_ = plt.subplots(1, 1, figsize=(10, 6))
    for dir_name in dirs:
        fig_, ax_ = plot_curve(dir_name, label=dir_name.split('_')[-1], fig=fig_, ax=ax_)

    return fig_, ax_


def compare_hist(dirs, labels=None, title=None, **kwargs):
    fig_, ax_ = plt.subplots(1, 1, figsize=(10, 6))
    if labels is None:
        labels = [_.split('_')[-1] for _ in dirs]

    if title is None:
        title = dirs[0].split('/')[1]
    for dir_name, lab in zip(dirs, labels):
        fig_, ax_ = plot_curve(dir_name, label=lab, fig=fig_, ax=ax_, title=title, **kwargs)

    return fig_, ax_


if __name__ == "__main__":
    fig, _ = compare_hist(["results/walker2d-medium-v2/DAC_b=1.0|QTar=lcb|rho=1.0"])
    fig.show()
