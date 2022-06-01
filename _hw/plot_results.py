"""
plot the results of training predictors
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _hw.utils as u
from matplotlib.transforms import Affine2D
from _hw.fit_lookup_table import get_metric


def plot_best_ds(df: pd.DataFrame, df_name: str, ds: str = None, metric="kendalltau", differentiable=True, tabular_value=None):
    plt.close("all")
    if isinstance(ds, str):
        df = df[df['dataset'] == ds]
        title = "Predictor performance on %s: %s" % (u.df_names[df_name], ds)
    else:
        ds = "mean"
        title = "Average over %s datasets" % u.df_names[df_name]

    if differentiable:
        df = df[df['predictor'].isin(u.diff_predictors)]

    all_predictors = df['predictor'].dropna().unique()
    all_train_sizes = df['train_size'].dropna().unique()

    # averaged stats
    df_p = df.groupby("predictor").mean()
    df_p = df_p[[metric, "train_time", "fit_time", "query_time"]]
    print(df_p.to_latex())

    # mixed data, train size curves
    df_p = df.groupby(["predictor", "train_size"]).mean().reset_index()
    df_p_std = df.groupby(["predictor", "train_size"]).std().reset_index()
    mean = df.groupby("train_size").mean()[metric]
    # std = df.groupby("train_size").std()[metric]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    for predictor in u.plot_infos.keys():
        if predictor in all_predictors:
            info = u.plot_infos[predictor]
            y = df_p[df_p['predictor'] == predictor][metric]
            ax1.plot(all_train_sizes, y, linestyle=info.line, label=info.name, c=info.col)
            ax2.plot(all_train_sizes, y.values-mean.values, linestyle=info.line, c=info.col)

    # maybe tabular lookup model, constant over dataset size
    if isinstance(tabular_value, float):
        info = u.plot_infos["tabular"]
        ax1.plot(all_train_sizes, [tabular_value for _ in all_train_sizes], linestyle=info.line, label=info.name, c=info.col)

    # ax1.set_xticks(all_train_sizes)
    ax1.set_title(title)
    ax1.set_xlabel("training set size")
    ax1.set_ylabel("%s (absolute)" % u.metric_names[metric])
    ax1.set_xscale("log")
    ax1.grid()

    # ax2.set_xticks(all_train_sizes)
    ax2.set_title(title)
    ax2.set_xlabel("training set size")
    ax2.set_ylabel("%s (centered on average)" % u.metric_names[metric])
    ax2.set_xscale("log")
    # ax2.set_ylim((0.0, 0.22))
    ax2.grid()

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # fig.legend(lines, labels, bbox_to_anchor=(1.18, 0.8))
    fig.legend(lines, labels, loc="center right", borderaxespad=0.3, frameon=False)

    # fig.suptitle(title)
    # fig.tight_layout()
    plt.subplots_adjust(right=0.82, wspace=0.3)

    u.save_cur_plot("mixed_data", metric, "diff_%s" % str(differentiable), df_name, ds)
    # plt.show()

    # plot error bars
    plt.close('all')
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    all_train_sizes2 = np.array([3*i for i in range(len(all_train_sizes))])
    i = 0
    for predictor in u.plot_infos.keys():
        if predictor in all_predictors:
            info = u.plot_infos[predictor]
            y = df_p[df_p['predictor'] == predictor][metric]
            y_std = df_p_std[df_p_std['predictor'] == predictor][metric]
            ax.errorbar(all_train_sizes2+(0.15*i), y, yerr=y_std, linestyle=info.line, label=info.name, c=info.col)
            i += 1

    desired_labels = [str(int(s)) for s in all_train_sizes]
    ax.set_xticks(all_train_sizes2)
    ax.set_xticklabels(desired_labels)

    ax.set_title(title)
    ax.set_xlabel("training set size")
    ax.set_ylabel("%s (absolute)" % u.metric_names[metric])
    # plt.xscale("log")
    ax.grid()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # fig.legend(lines, labels, bbox_to_anchor=(1.18, 0.8))
    fig.legend(lines, labels, loc="lower right", borderaxespad=0.3)

    u.save_cur_plot("mixed_data", metric, "diff_%s" % str(differentiable), df_name, "%s_std" % ds)


def plot_best(df_name: str):
    metrics = ["kendalltau", "diff_std", "fit_time"]  # ["kendalltau", "diff_std", "train_time", "fit_time", "query_time"]
    df = u.get_result_data(df_name)

    # lin reg has some severe outliers, which also drag down the KT at that time
    # most likely cause: the training data did not contain some operations at all
    df[df['diff_std'] > 2] = None

    for m in metrics:
        all_datasets = df['dataset'].dropna().unique()

        # tabular lookup
        tabular_values = {}
        for ds in all_datasets:
            v = get_metric(df_name, ds, m)  # get lookup table performance
            if v is None:
                break
            tabular_values[ds] = v
        if len(tabular_values) > 0:
            tabular_values['mean'] = sum(list(tabular_values.values())) / len(tabular_values)

        # plot
        diff = False
        plot_best_ds(df, df_name, None, differentiable=diff, metric=m, tabular_value=tabular_values.get('mean', None))
        for ds in all_datasets:
            plot_best_ds(df, df_name, ds, differentiable=diff, metric=m, tabular_value=tabular_values.get(ds, None))


if __name__ == '__main__':
    plot_best("results_transnas")
    plot_best("results_hwnas")
