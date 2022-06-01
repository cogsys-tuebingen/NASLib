"""
plot some correlations between predictor results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import _hw.utils as u


def plot_corr(df: pd.DataFrame, df_name: str, metric1="kendalltau", metric2="diff_std"):
    plt.close("all")

    all_datasets = df['dataset'].dropna().unique()
    # all_predictors = df['predictor'].dropna().unique()
    all_train_sizes = df['train_size'].dropna().unique()
    ds = "mean"

    # only interesting stats
    df = df.dropna()
    df = df[df['train_size'] == all_train_sizes[-1]]

    # plot
    m1 = df[metric1].values
    m2 = df[metric2].values

    corr_p = np.abs(np.corrcoef(m1, m2)[1, 0])
    corr_s = stats.spearmanr(m1, m2)[0]
    corr_k = stats.kendalltau(m1, m2)[0]

    plt.scatter(m1, m2, label="KT=%.2f, SCC=%.2f, PCC=%.2f" % (corr_k, corr_s, corr_p), s=15)
    # min_, max_ = np.min(ytest), np.max(ytest)
    # plt.plot([min_, max_], [min_, max_], "r-")
    plt.xlabel(u.metric_names[metric1])
    plt.ylabel(u.metric_names[metric2])
    # plt.title("Predictions and targets")
    plt.legend()
    plt.grid()

    u.save_cur_plot("metric_corr", "%s-%s" % (metric1, metric2), df_name, ds)
    # plt.show()


def plot_multi_corr(df_name: str):
    df = u.get_result_data(df_name)
    df[df['diff_std'] > 2] = None  # lin reg has some severe outliers, which also drag down the KT at that time

    plot_corr(df, df_name, metric1="diff_std", metric2="kendalltau")


if __name__ == '__main__':
    plot_multi_corr("results_transnas")
    plot_multi_corr("results_hwnas")
