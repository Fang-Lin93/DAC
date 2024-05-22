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


if __name__ == '__main__':
    res_dirs = {}

    res_dirs['antmaze-large-play-v0'] = [
        "results/antmaze-large-play-v0/20240127-145215_DAC(eta=0.1)maxminQ",
        "results/antmaze-large-play-v0/20240127-222234_DAC(eta=1.0)maxminQ",
    ]

    res_dirs['hopper-medium-expert-v2'] = [
        "results/hopper-medium-expert-v2/20231206-144201_dbc",
        "results/hopper-medium-expert-v2/20231130-080222_iql",
        "results/hopper-medium-expert-v2/20231130-091823_ivr",
        # "results/hopper-medium-expert-v2/20231222-212024_dql",

        # "results/hopper-medium-expert-v2/20240111-072445_dpieta01",  # best
        # "results/hopper-medium-expert-v2/20240112-232144_dpieta10RA10",
        # "results/hopper-medium-expert-v2/20240113-130015_dpieta01RA1",
        # "results/hopper-medium-expert-v2/20240113-165855_dpieta1RA1",
        # "results/hopper-medium-expert-v2/20240113-204726_dpieta1RA10",

        # "results/hopper-medium-expert-v2/20240114-002850_dpieta5RA10",
        # "results/hopper-medium-expert-v2/20240114-094315_dpieta10RA10",
        # "results/hopper-medium-expert-v2/20240114-141758_dpieta15RA10",
        # "results/hopper-medium-expert-v2/20240114-180710_dpieta10RA20",
        # "results/hopper-medium-expert-v2/20240114-203949_dpieta10RA10Re",
        # "results/hopper-medium-expert-v2/20240114-233927_dpieta10RA10Ree",

        # "results/hopper-medium-expert-v2/20240115-092410_dpieta10RA10adv",
        # "results/hopper-medium-expert-v2/20240115-132638_dpieta10RA1new",

        # "results/hopper-medium-expert-v2/20240116-083140_dpieta10",
        # "results/hopper-medium-expert-v2/20240116-083306_dpieta10+RA10",
        # "results/hopper-medium-expert-v2/20240116-083028_dpieta10+lambda",
        # "results/hopper-medium-expert-v2/20240116-082854_dpieta10+RA10+lambda",

        # "results/hopper-medium-expert-v2/20240116-153840_dpieta10",
        # "results/hopper-medium-expert-v2/20240116-205650_dpieta10+lambda",
        # "results/hopper-medium-expert-v2/20240117-205959_dpieta100+RA50",
        # "results/hopper-medium-expert-v2/20240118-071124_dpieta10+RA10+lambda01",

        # "results/hopper-medium-expert-v2/20240118-145453_dpieta10+RA10+lambda001",
        #
        # "results/hopper-medium-expert-v2/20240116-083306_dpieta10+RA10",
        # "results/hopper-medium-expert-v2/20240119-132221_dpiT5eta10RA10+2Qs",
        # "results/hopper-medium-expert-v2/20240119-154607_dpiT5eta10RA10+5Qs",

        "results/hopper-medium-expert-v2/20240123-014111_dpieta001",
        "results/hopper-medium-expert-v2/20240123-014234_dpieta01",
        # "results/hopper-medium-expert-v2/20240123-102640_dpieta1",  # best
        # "results/hopper-medium-expert-v2/20240123-102734_dpieta10",

        # "results/hopper-medium-expert-v2/20240125-153937_dpieta1tune",
        # "results/hopper-medium-expert-v2/20240128-001619_DAC(eta=0.1)",
        "results/hopper-medium-expert-v2/20240128-135940_DAC(eta=0.1)minE",
        "results/hopper-medium-expert-v2/20240128-171551_DAC(eta=0.01)minE",
    ]

    # hopper-medium-v2
    res_dirs['hopper-medium-v2'] = [

        # SOTA
        # "results/hopper-medium-v2/20231209-055233_dbc",
        "results/hopper-medium-v2/20231208-211552_iql",
        "results/hopper-medium-v2/20231209-011311_ivr",
        "results/hopper-medium-v2/20231221-023639_dql",
        "results/hopper-medium-v2/20240126-111410_DAC(eta=0.005)",
        "results/hopper-medium-v2/20240128-190658_DAC(eta=0.005)",
        "results/hopper-medium-v2/20240131-234243_DAC(bc=2.0)DualClip",
        # "results/hopper-medium-v2/20240201-105730_DAC(bc=1.0)LCB",
        "results/hopper-medium-v2/20240201-142009_DAC(bc=1.0)bdLCB",  # Try BC =2
        "results/hopper-medium-v2/20240214-120114_DAC_QTar_tscs",
    ]

    res_dirs['hopper-medium-replay-v2'] = [
        "results/hopper-medium-replay-v2/20231211-100601_bc",
        "results/hopper-medium-replay-v2/20231211-104224_iql",
        "results/hopper-medium-replay-v2/20231211-114933_ivr",
        "results/hopper-medium-replay-v2/20231211-125456_dbc",
        "results/hopper-medium-replay-v2/20231221-122228_dql",

        # "results/hopper-medium-replay-v2/20240112-233538_dpieta10",
        # "results/hopper-medium-replay-v2/20240114-005628_dpieta5RA10",
        # "results/hopper-medium-replay-v2/20240114-100222_dpieta10RA10",
        # "results/hopper-medium-replay-v2/20240114-141730_dpieta15RA10",
        # "results/hopper-medium-replay-v2/20240114-233939_dpieta10RA10Ree",

        # "results/hopper-medium-replay-v2/20240115-092420_dpieta10RA10adv",
        # "results/hopper-medium-replay-v2/20240115-132836_dpieta10RA1new",
        # "results/hopper-medium-replay-v2/20240118-145733_dpieta10+RA10+lambda001",

        # "results/hopper-medium-replay-v2/20240123-144820_dpieta1",
        # "results/hopper-medium-replay-v2/20240123-160753_dpieta01",  # best

        # "results/hopper-medium-replay-v2/20240125-155850_dpieta01tune",
        "results/hopper-medium-replay-v2/20240126-201658_DAC(eta=0.01)",
        "results/hopper-medium-replay-v2/20240127-220112_DAC(eta=0.01)",
        # "results/hopper-medium-replay-v2/20240128-005416_DAC(eta=0.1)",
        "results/hopper-medium-replay-v2/20240123-160753_dpieta01",  # best
        "results/hopper-medium-replay-v2/20240128-140005_DAC(eta=0.1)minE"
    ]
    # hopper-medium-v2  hopper-medium-replay-v2 hopper-medium-expert-v2
    data = res_dirs['hopper-medium-v2']  #
    # data = ["results/antmaze-medium-play-v0/20240126-014644_DAC(eta=0.1)",
    #          ]

    data = [
        "results/halfcheetah-medium-v2/20240130-020557_DAC(eta=0.01)Dual",
        "results/halfcheetah-medium-v2/20240205-115631_DAC(bc=1.0, eta>=0.01)best",
        "results/halfcheetah-medium-v2/20240216-203447_DAC_QTar_ts01std",
        "results/halfcheetah-medium-v2/20240219-135620_DAC_QTar_tstd",
        "results/halfcheetah-medium-v2/20240219-173835_DAC_QTar_tsVectd",

    ]
    #
    data = [
        # "results/walker2d-medium-v2/20240206-100601_DAC(bc=1.0-eta=0.01+0.1minE)",
        # "results/walker2d-medium-v2/20240205-115607_DAC(bc=1.0, eta>=0.01)best",
        # "results/walker2d-medium-v2/20240213-221639_DAC_QTar_minE",
        # "results/walker2d-medium-v2/20240214-002224_DAC_QTar_minEfixedeta",
        # "results/walker2d-medium-v2/20240214-120200_DAC_QTar_tscs",
        # "results/walker2d-medium-v2/20240215-123852_DAC_QTar_csbc05",
        # "results/walker2d-medium-v2/20240215-123924_DAC_QTar_cseta0001",
        # "results/walker2d-medium-v2/20240215-175944_DAC_QTar_csema5Q",  # also + eta0001
        # "results/walker2d-medium-v2/20240215-230754_DAC_QTar_minEeta0001",
        # "results/walker2d-medium-v2/20240215-235633_DAC_QTar_ts2std",
        # "results/walker2d-medium-v2/20240216-085922_DAC_QTar_minEmaxstdLarge",  # large positive std penalty
        # "results/walker2d-medium-v2/20240216-092730_DAC_QTar_tsmaxstd",  # negative std!
        # "results/walker2d-medium-v2/20240216-133046_DAC_QTar_tsmaxstdLarge",
        # "results/walker2d-medium-v2/20240216-164406_DAC_QTar_csmax1std",
        # "results/walker2d-medium-v2/20240217-170433_DAC_QTar_cs001std001eta",
        "results/walker2d-medium-v2/20240218-183252_DAC_QTar_minE",
        # "results/walker2d-medium-v2/20240217-191923_DAC_QTar_csbase",
        # "results/walker2d-medium-v2/20240219-135833_DAC_QTar_tstd",
        # "results/walker2d-medium-v2/20240219-163149_DAC_QTar_tsVectd",
        "results/walker2d-medium-v2/20240219-184047_DAC_QTar_csDAC",
        # "results/walker2d-medium-v2/20240220-115405_DAC_QTar_csVar"
        "results/walker2d-medium-v2/20240220-163831_DAC_QTar_minEFixedEta",
        "results/hopper-medium-v2/20240220-180909_DAC_QTar_csFixedEta001",
        "results/walker2d-medium-v2/20240220-231754_DAC_QTar_csFixedEta",

    ]

    # data = [
    #     "results/walker2d-medium-replay-v2/20240214-181403_DAC_QTar_minE",
    #     "results/walker2d-medium-replay-v2/20240205-161414_DAC(bc=1.0, eta>=0.01)best",
    #     "results/walker2d-medium-replay-v2/20240215-170333_DAC_QTar_cseta0001",
    #
    #     ]
    # data = [
    #
    #     # SOTA
    #     "results/hopper-medium-v2/20231208-211552_iql",
    #     "results/hopper-medium-v2/20231209-011311_ivr",
    #     "results/hopper-medium-v2/20231221-023639_dql",
    #     "results/hopper-medium-v2/20240131-110018_DAC(eta=0.01)DualClip",
    #     "results/hopper-medium-v2/20240205-115419_DAC(bc=1.0, eta>=0.01)best",
    #     "results/hopper-medium-v2/20240215-165503_DAC_QTar_cseta0001",
    #     "results/hopper-medium-v2/20240216-084151_DAC_QTar_tsmaxstd",
    #     "results/hopper-medium-v2/20240216-133105_DAC_QTar_tsmaxstdLarge",  # large negative std!
    #     "results/hopper-medium-v2/20240217-191929_DAC_QTar_csbase",
    #     # "results/hopper-medium-v2/20240217-170533_DAC_QTar_cs001std001eta",
    #
    # ]

    data = [
        "results/hopper-medium-v2/20231208-211552_iql",
        "results/hopper-medium-v2/20231209-011311_ivr",
        "results/hopper-medium-v2/20231221-023639_dql",
        "results/hopper-medium-v2/20240217-191929_DAC_QTar_csbase",
        # "results/hopper-medium-v2/20240218-080348_DAC_QTar_minE2Qheads",
        "results/hopper-medium-v2/20240218-181636_DAC_QTar_minE",
        # "results/hopper-medium-v2/20240219-135858_DAC_QTar_tstd",
        # "results/hopper-medium-v2/20240219-184342_DAC_QTar_csDAC",
        # "results/hopper-medium-v2/20240220-115306_DAC_QTar_csVar",
        "results/hopper-medium-v2/20240126-111410_DAC(eta=0.005)",
        "results/hopper-medium-v2/20240220-175228_DAC_QTar_csDAC2bc",
        "results/hopper-medium-v2/20240220-180909_DAC_QTar_csFixedEta001",
        "results/hopper-medium-v2/20240220-231733_DAC_QTar_csFixedEta",
        # "results/hopper-medium-v2/20240220-133643_DAC_QTar_minEVarFixedEta0005",
        # "results/hopper-medium-v2/20240220-164039_DAC_QTar_minEFixedEta",
    ]

    ########################################################################

    data = [
        "results/halfcheetah-medium-v2/20240205-115631_DAC(bc=1.0, eta>=0.01)best",
        "results/halfcheetah-medium-v2/20240223-104331_DAC_QTar=lcb+eta=0.005",
        "results/halfcheetah-medium-v2/20240328-142744_DACBC<=1.0+QTar=lcbNoClip"
    ]

    data = [
        # "results/hopper-medium-v2/20231208-211552_iql",
        # "results/hopper-medium-v2/20231209-011311_ivr",
        "results/hopper-medium-v2/20231221-023639_dql",
        # "results/hopper-medium-v2/20240217-191929_DAC_QTar_csbase",
        # "results/hopper-medium-v2/20240218-181636_DAC_QTar_minE",
        # "results/hopper-medium-v2/20240223-104346_DAC_QTar=lcb+eta=0.005",
        # "results/hopper-medium-v2/20240223-145817_DAC_QTar=lcb+BC<=1.0rho2",
        # "results/hopper-medium-v2/20240223-170040_DAC_QTar=lcb+BC<=1.0rho10",  # eta_lr = 1
        # "results/hopper-medium-v2/20240327-225024_DACBC<=1.0+QTar=lcbNoClip",
        # "results/hopper-medium-v2/20240328-222204_DACBC<=0.8+QTar=lcbNoClip",
        "results/hopper-medium-v2/20240329-142958_DACBC<=0.5+QTar=lcbNoClip",
        # "results/hopper-medium-v2/20240331-180530_DACBC<=0.2+QTar=lcbNoClip"
        # "results/hopper-medium-v2/20240401-101843_DACBC<=0.5+QTar=Elcb",
        # "results/hopper-medium-v2/20240401-142318_DACBC<=1.0+QTar=Elcb",
        # "results/hopper-medium-v2/20240401-164202_DACBC<=1.0+QTar=Elcbmineta",
        # "results/hopper-medium-v2/20240402-072020_DACBC<=0.25+QTar=lcbmineta",
        # "results/hopper-medium-v2/20240402-125312_DACBC<=0.25+QTar=lcbminetaRM2",
        # "results/hopper-medium-v2/20240402-125513_DACBC<=0.25+QTar=ElcbminetaRM2",
        # "results/hopper-medium-v2/20240403-192106_DACBC<=0.5+QTar=lcbminetaRM2",
        # "results/hopper-medium-v2/20240404-091608_DACBC<=0.2+QTar=lcbminetaRM2",
        # "results/hopper-medium-v2/20240405-153301_DACBC<=0.5+QTar=lcbLagecyRM2",
        "results/hopper-medium-v2/20240407-232338_DACBC<=1.0+QTar=lcb10Q",
        "results/hopper-medium-v2/20240408-094038_DACBC<=0.5+QTar=lcb10Q",
        "results/hopper-medium-v2/20240408-222646_DACBC<=0.25+QTar=lcb10Q"
    ]

    data = [
        # "results/walker2d-medium-v2/20240326-142711_DACBC<=1.0+QTar=minE",
        "results/walker2d-medium-v2/20240218-183252_DAC_QTar_minE",
        "results/walker2d-medium-v2/20240219-184047_DAC_QTar_csDAC",
        "results/walker2d-medium-v2/20240223-104320_DAC_QTar=lcb+eta=0.005",
        "results/walker2d-medium-v2/20240223-143711_DAC_QTar=minE+eta=0.005",
        "results/walker2d-medium-v2/20240321-212053_DAC_QTar=td+BC<=1.0",
        "results/walker2d-medium-v2/20240326-142711_DACBC<=1.0+QTar=minE",
        "results/walker2d-medium-v2/20240327-225021_DACBC<=1.0+QTar=lcbNoClip",
        # "results/walker2d-medium-v2/20240331-142021_DACBC<=0.2+QTar=lcbNoClip",
        # "results/walker2d-medium-v2/20240404-143926_DACBC<=0.25+QTar=lcbminetaRM2",
        # "results/walker2d-medium-v2/20240404-232911_DACBC<=0.5+QTar=lcbminetaRM2",
        # "results/walker2d-medium-v2/20240405-095757_DACBC<=0.5+QTar=lcbRM2",
        # "results/walker2d-medium-v2/20240406-061528_DACBC<=0.5+QTar=lcbRM2ResNet",
    ]

    # ablation study to find the best configuration
    data = [
        "results/walker2d-medium-v2/20240406-092038_DACBC<=1.0+QTar=lcbraw",
        "results/walker2d-medium-v2/20240406-133026_DACBC<=1.0+QTar=lcbhp",
        "results/walker2d-medium-v2/20240406-173344_DACBC<=1.0+QTar=lcb10Q",
        "results/walker2d-medium-v2/20240406-230819_DACBC<=1.0+QTar=lcbresnet",
        "results/walker2d-medium-v2/20240407-124130_DACBC<=1.0+QTar=lcb10Q",
        # "results/walker2d-medium-v2/20240409-112717_DACBC<=1.0+QTar=eptrho08",  # action level ept
        "results/walker2d-medium-v2/20240409-150051_DACBC<=1.0+QTar=eptrho09",
        # "results/walker2d-medium-v2/20240409-171231_DACBC<=1.0+QTar=lcb10Q",
        "results/walker2d-medium-v2/20240409-224704_DACBC<=1.0+QTar=lcb5Q",
        "results/walker2d-medium-v2/20240410-000247_DACBC<=1.0+QTar=lcb10Q",
        "results/walker2d-medium-v2/20240410-082345_DACBC<=0.5+QTar=lcb10Q",
        # "results/walker2d-medium-v2/20240410-101650_DACBC<=0.25+QTar=eptrho08",
        "results/walker2d-medium-v2/20240410-141048_DACBC<=1.0+QTar=lcbQ10"
    ]

    ############ self comparison

    data  = [
        "results/halfcheetah-medium-v2/20240421-214029_DACBC<=1.0+QTar=lcbrho0",
        "results/halfcheetah-medium-v2/20240422-161536_DACBC<=2.0+QTar=lcbrho0"
    ]

    data = [
        # "results/hopper-medium-v2/20240411-062755_DACBC<=1.0+QTar=lcbQ10",
        # "results/hopper-medium-v2/20240413-144054_DACBC<=0.5+QTar=lcbQ10",
        "results/hopper-medium-v2/20240415-141022_DACBC<=1.0+QTar=lcbQ10",
        "results/hopper-medium-v2/20240418-025302_DACBC<=0.5+QTar=lcbQ10",
        "results/hopper-medium-v2/20240416-150007_DACBC<=1.0+QTar=minEQ10",
        "results/hopper-medium-v2/20240422-161027_DACBC<=1.0+QTar=lcbrho15",
        "results/hopper-medium-v2/20240425-091821_DACBC<=1.0+QTar=lcbrho12",
        "results/hopper-medium-v2/20240511-073230_DACBC<=1.0|QTar=lcb|rho=1.5|sota"
    ]

    data = [
        "results/walker2d-medium-v2/20240410-141048_DACBC<=1.0+QTar=lcbQ10",
        "results/walker2d-medium-v2/20240412-223348_DACBC<=0.5+QTar=lcbQ10",
        "results/walker2d-medium-v2/20240414-173429_DACBC<=1.0+QTar=lcbmeanQAct",
        "results/walker2d-medium-v2/20240417-004640_DACBC<=1.0+QTar=minEQ10",
    ]

    data = [
        # "results/antmaze-umaze-v0/20240427-143919_DACBC<=1.0+QTar=lcb",
        # "results/antmaze-umaze-v0/20240428-090259_DACBC<=1.0|QTar=lcb|rho=1.0|base",
        # "results/antmaze-umaze-v0/20240428-115615_DACBC<=1.0|QTar=lcb|rho=1.5|base",
        # "results/antmaze-umaze-v0/20240505-070335_DACBC<=0.01|QTar=lcb|rho=1.0|bc001",
        # "results/antmaze-umaze-v0/20240505-105118_DACBC<=0.1|QTar=lcb|rho=1.0|bc01",  # raw code
        # "results/antmaze-umaze-v0/20240505-112511_DACBC<=0.1|QTar=lcb|rho=1.0|bc01",  # add online code
        "results/antmaze-umaze-v0/20240505-112511_DACBC<=0.1|QTar=lcb|rho=1.0|bc01",
        "results/antmaze-umaze-v0/20240508-181112_DACBC<=0.1|QTar=lcb|rho=1.0|Dual",
        "results/antmaze-umaze-v0/20240508-211920_DACBC<=0.5|QTar=lcb|rho=1.0|Dual",

        # "results/antmaze-medium-play-v0/20240501-150841_DACBC<=1.0|QTar=lcb|rho=1.0|loco",
        # "results/antmaze-medium-play-v0/20240503-032038_DACeta=1.0|QTar=lcb|rho=1.0|loco",
        # "results/antmaze-medium-play-v0/20240504-084342_DACeta=0.1|QTar=lcb|rho=1.0|loco",
        # "results/antmaze-umaze-diverse-v0/20240505-135042_DACBC<=0.1|QTar=lcb|rho=1.0|bc01",
        "results/antmaze-umaze-diverse-v0/20240501-113235_DACeta=0.1|QTar=lcb|rho=1.0|loco",
        # "results/antmaze-umaze-diverse-v0/20240505-171110_DACBC<=0.1|QTar=lcb|rho=1.0|CQLbc01",
        # "results/antmaze-umaze-diverse-v0/20240501-113235_DACeta=0.1|QTar=lcb|rho=1.0|loco",
        # "results/antmaze-umaze-diverse-v0/20240504-000511_DACeta=0.2|QTar=lcb|rho=1.0|loco",
        # "results/antmaze-umaze-diverse-v0/20240504-035454_DACeta=0.5|QTar=lcb|rho=1.0|loco",
        "results/antmaze-umaze-diverse-v0/20240505-201120_DACeta=0.1|QTar=lcb|rho=1.0|ETA",
        "results/antmaze-umaze-diverse-v0/20240506-000300_DACeta=0.1|QTar=lcb|rho=0.5|ETA",
        "results/antmaze-umaze-diverse-v0/20240508-235916_DACBC<=0.5|QTar=lcb|rho=1.0|Dual"


    ]

    data = [
        "results/antmaze-large-play-v0/20240504-163818_DACeta=0.1|QTar=lcb|rho=1.0|loco",
        "results/antmaze-large-play-v0/20240503-121030_DACeta=1.0|QTar=lcb|rho=1.0|loco",
        "results/antmaze-large-play-v0/20240506-111505_DACeta=0.5|QTar=lcb|rho=1.0|ETA",
        "results/antmaze-large-play-v0/20240507-023155_DACeta=0.8|QTar=lcb|rho=1.0|ETA",
        "results/antmaze-large-play-v0/20240507-153343_DACeta=1.0|QTar=lcb|rho=1.0|ETA",
        "results/antmaze-large-play-v0/20240508-050425_DACeta=1.5|QTar=lcb|rho=1.0|ETA",
        "results/antmaze-large-play-v0/20240508-102455_DACeta=0.8|QTar=lcb|rho=0.5|ETA",

        "results/antmaze-large-diverse-v0/20240506-173411_DACeta=0.5|QTar=lcb|rho=1.0|ETA",
        "results/antmaze-large-diverse-v0/20240507-092550_DACeta=0.8|QTar=lcb|rho=1.0|ETA",
        "results/antmaze-large-diverse-v0/20240507-223602_DACeta=1.0|QTar=lcb|rho=1.0|ETA",

    ]

    # data = [
    #     "results/walker2d-medium-v2/20240414-173429_DACBC<=1.0+QTar=lcbmeanQAct",
    #     "results/walker2d-medium-v2/20240509-151417_DACBC<=1.0|QTar=lcb|rho=1.0|sota",
    #     "results/walker2d-medium-v2/20240509-200813_DACBC<=1.0|QTar=lcb|rho=1.0|sota",
    #     "results/walker2d-medium-v2/20240509-232450_DACBC<=1.0|QTar=lcb|rho=1.0|sota",
    #     "results/walker2d-medium-v2/20240510-013301_DACBC<=1.0|QTar=lcb|rho=1.0|sota",
    #     "results/walker2d-medium-v2/20240510-153803_DACBC<=1.0|QTar=lcb|rho=1.0|sota"  # use scanning buffer!
    # ]

    data = [

        "results/antmaze-umaze-diverse-v0/20240501-113235_DACeta=0.1|QTar=lcb|rho=1.0|loco",
        "results/antmaze-umaze-diverse-v0/20240512-201837_DACeta=0.1|QTar=lcb|rho=1.0|sota",
        # "results/antmaze-umaze-diverse-v0/20240514-085804_DACBC<=0.08|QTar=lcb|rho=1.0|RewardTuned",
        # "results/antmaze-umaze-diverse-v0/20240514-103200_DACeta=0.1|QTar=lcb|rho=1.0|RewardTuned",
        # "results/antmaze-umaze-diverse-v0/20240514-152304_DACeta=0.05|QTar=lcb|rho=1.2|fixed",
        # "results/antmaze-umaze-diverse-v0/20240514-231303_DACeta=0.05|QTar=lcb|rho=1.0|fixed",
        # "results/antmaze-umaze-diverse-v0/20240515-222154_DACeta=0.05|QTar=lcb|rho=1.1|fixedLRrho",
        "results/antmaze-umaze-diverse-v0/20240516-140358_DACeta=0.1|QTar=lcb|rho=1.0|same",
        "results/antmaze-umaze-diverse-v0/20240516-161857_DACeta=0.1|QTar=lcb|rho=1.0|diff",
        "results/antmaze-umaze-diverse-v0/20240517-143459_DACeta=0.1|QTar=lcb|rho=1.0|samelr",
        "results/antmaze-umaze-diverse-v0/20240517-224721_DACeta=0.1|QTar=lcb|rho=1.0|Last"
    ]

    data = [
        "results/antmaze-umaze-v0/20240517-083113_DACeta=0.02|QTar=lcb|rho=1.0|same",
        "results/antmaze-umaze-v0/20240515-033731_DACeta=0.05|QTar=lcb|rho=1.0|fixed",
        "results/antmaze-umaze-v0/20240517-184317_DACeta=0.1|QTar=lcb|rho=1.0|Last",
        "results/antmaze-umaze-v0/20240518-065433_DACeta=0.1|QTar=lcb|rho=1.0|Last",  # wo shuffle!

    ]

    data = [
        # "results/antmaze-umaze-diverse-v0/20240517-224721_DACeta=0.1|QTar=lcb|rho=1.0|Last",
        # "results/antmaze-umaze-diverse-v0/20240518-123235_DACeta=0.1|QTar=lcb|rho=1.0|Lasttune"
        # "results/antmaze-large-diverse-v0/20240506-173411_DACeta=0.5|QTar=lcb|rho=1.0|ETA",
        # "results/antmaze-large-diverse-v0/20240507-092550_DACeta=0.8|QTar=lcb|rho=1.0|ETA",
        # "results/antmaze-large-diverse-v0/20240507-223602_DACeta=1.0|QTar=lcb|rho=1.0|ETA",
        "results/antmaze-large-diverse-v0/20240518-152718_DACeta=0.8|QTar=lcb|rho=1.0|Last",
        "results/antmaze-large-diverse-v0/20240518-204734_DACeta=0.1|QTar=lcb|rho=1.0|Last",
        "results/antmaze-large-diverse-v0/20240519-014350_DACeta=0.5|QTar=lcb|rho=1.0|Last",
        "results/antmaze-large-diverse-v0/20240519-044425_DACeta=0.05|QTar=lcb|rho=1.0|Last",
        "results/antmaze-large-diverse-v0/20240519-095843_DACeta=1.0|QTar=lcb|rho=1.0|Last"
    ]

    data = [
        "results/walker2d-medium-v2/20240518-151950_DACBC<=1.0|QTar=lcb|rho=1.0|hard",  # 85.24
        "results/walker2d-medium-v2/20240510-153803_DACBC<=1.0|QTar=lcb|rho=1.0|sota",  # 95.8

        "results/walker2d-medium-replay-v2/20240518-193112_DACBC<=1.0|QTar=lcb|rho=1.0|hard",  # 96.9
        "results/walker2d-medium-replay-v2/20240510-204512_DACBC<=1.0|QTar=lcb|rho=1.0|sota",  # 96.8

        "results/walker2d-medium-expert-v2/20240511-023638_DACBC<=1.0|QTar=lcb|rho=1.0|sota",  # 110.4
        "results/walker2d-medium-expert-v2/20240519-003949_DACBC<=1.0|QTar=lcb|rho=1.0|hard"  # 111.6
    ]

    data = [
        "results/hopper-medium-v2/20240511-073230_DACBC<=1.0|QTar=lcb|rho=1.5|sota",  # 103.1
        "results/hopper-medium-v2/20240519-045514_DACBC<=1.0|QTar=lcb|rho=1.5|hard"  # 101.2
        
        "results/hopper-medium-replay-v2/20240511-124434_DACBC<=1.0|QTar=lcb|rho=1.5|sota",  # 103.8
        "results/hopper-medium-replay-v2/20240519-092631_DACBC<=1.0|QTar=lcb|rho=1.5|hard",  # 103.1
    ]

    f, a = compare_hist(data, curve='mean', window_size=1, labels=None, title=None)
    f.show()
