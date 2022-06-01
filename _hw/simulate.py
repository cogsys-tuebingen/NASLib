import os
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import Iterable
from pymoo.performance_indicator.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import _hw.utils as u


rng = np.random.default_rng()
nds = NonDominatedSorting()


def get_hypervolume(data: np.array) -> Hypervolume:
    assert len(data.shape) == 2
    assert data[0, 0] <= 0
    return Hypervolume(ref_point=(0, 1.1*max(data[:, 1])))


def get_noise(std: float, size: int) -> np.array:
    """
    weighted mix of 3 distributions:
        N(0, std*norm_mul1)     moved slightly to the positive
        N(0, std*norm_mul2)     moved slightly to the negative
        U(-std*uniform_mul, std*uniform_mul)
    """
    # defaults
    uni_prob = 0.01
    norm_prob1 = 0.725
    norm_prob2 = 1 - norm_prob1 - uni_prob
    norm_mul1 = 1
    norm_mul2 = 3
    uniform_mul = 15

    # random parametrization
    shift = rng.uniform(0, 2*std)

    # less smooth
    repeats = 15
    size2 = size // repeats

    # random distributions
    r = rng.random(size=size2)
    n1 = (norm_mul1 * rng.standard_normal(size2) + shift) * (r < norm_prob1)
    r -= norm_prob1
    n2 = (norm_mul2 * rng.standard_normal(size2) - shift) * (r < norm_prob2) * (r > 0)
    r -= norm_prob2
    u1 = rng.uniform(-uniform_mul, uniform_mul, size=size2) * (r > 0)

    # put together, make sure the std matches what we want, match size
    mixed = np.repeat(n1 + n2 + u1, repeats)
    if len(mixed) < size:
        mixed = np.concatenate([mixed, mixed[:size-len(mixed)]])
    mixed = mixed * std / np.std(mixed)
    return mixed


def get_noisy_best(subset: np.array, acc_noise=0.0, hw_noise=0.0) -> (np.array, float):
    """
    get indices of the best architectures in the subset and their kt+hypervolume, after adding noise to the metrics
    """
    # possibly add noise to the hardware metric
    if acc_noise > 0 or hw_noise > 0:
        noisy_subset = np.copy(subset)
        if acc_noise > 0:
            noisy_subset[:, 0] += get_noise(acc_noise, subset.shape[0])
        if hw_noise > 0:
            noisy_subset[:, 1] += get_noise(hw_noise, subset.shape[0])
    else:
        noisy_subset = subset

    # select the best architectures according to the (noisy) metrics
    sup_best_idx_ = nds.do(noisy_subset, only_non_dominated_front=True)

    # but use their ground truth values to compare
    best_actually = subset[sup_best_idx_]
    best_supposed = noisy_subset[sup_best_idx_]

    # calc hypervolume on true acc/hw values
    hv = get_hypervolume(subset).calc(best_actually)

    # pareto front of supposed best
    sup_best_idx_ = nds.do(best_actually, only_non_dominated_front=True)
    best_actually_p = best_actually[sup_best_idx_]

    return {
        'actually_best': best_actually,
        'supposedly_best': best_supposed,
        'actually_best_pareto': best_actually_p,
        'kt': stats.kendalltau(subset[:, 1], noisy_subset[:, 1])[0],  # correlation of all predicted/true metric values
        'hv': hv,
    }


def collect_sim_data(hv_csv_path: str, start_trial=0, end_trial=100) -> pd.DataFrame:
    """
    simulate the selection of architectures, calculate some metrics
    """
    # settings
    num_architectures = [100, 500, 1000, 2000, 5000, 15625]
    noise_values = [v * 0.1 for v in range(11)]
    acc_noise_values = [0.0]
    do_plot = False

    # for scatter plots
    num_architectures = [15625]
    noise_values = [0.0, 0.5]
    acc_noise_values = [0.0]
    start_trial, end_trial = 0, 5
    do_plot = True

    # set up....
    num_arcs = 15625
    arc_indices = np.array(range(15625))
    full_df = pd.read_csv(hv_csv_path, index_col=0) if os.path.isfile(hv_csv_path) else None
    columns = ('dataset', 'metric', 'num_arc', 'trial', 'acc std',
               'hv for hw std: [%s]' % ", ".join(["%.1f" % n for n in noise_values]),
               'kt', 'acc_all_to_par', 'acc_par_to_par', 'hw_all_to_par', 'hw_par_to_par')
    t0 = time.time()

    # load and sort hwnas
    hwnas_data = u.get_hwnas_data()
    hw_col0 = hwnas_data.columns[0]
    # all_hw_datasets = hwnas_data[hw_col0].unique()
    all_hw_datasets = u.hard_datasets_hwnas
    hwnas_data = hwnas_data[sorted(hwnas_data.columns)]  # also contains datasets with -1 values

    # load and sort nb201, create lookup dict of sorted accuracy values
    nb201_data = u.get_nb201_data()
    nb_col0 = nb201_data.columns[0]
    all_nb_datasets = nb201_data[nb_col0].unique()
    del nb201_data[nb_col0]
    nb201_data = nb201_data[sorted(nb201_data.columns)]
    # minimization problem: cast accuracy to [0, 1] range, make negative
    nb201_lookup = {ds: nb201_data.iloc[i].to_numpy(copy=True) * (-0.01) for (i, ds) in enumerate(all_nb_datasets)}
    del nb201_data

    #
    assert 0.0 in noise_values
    t1 = time.time()
    print("loaded/sorted all data, in %.1f seconds" % (t1 - t0))

    # for each hwnas data set and gaussian noise value, sample 'num_trials'
    for i, ds_metric in enumerate(all_hw_datasets):
        ds, metric = ds_metric.rsplit('-', 1)

        # maybe skip, already have the data
        if isinstance(full_df, pd.DataFrame) and (not do_plot):
            expected_num = len(num_architectures) * (end_trial - start_trial) * len(acc_noise_values)
            df = full_df[(full_df['dataset'] == ds) & (full_df['metric'] == metric)]
            # missing something after loading, remove all and start over
            if len(df) < expected_num:
                full_df = full_df.drop(full_df[(full_df['dataset'] == ds) & (full_df['metric'] == metric)].index)
            else:
                print("data for [%s] is available, skipping" % ds_metric)
                continue

        # load normalized values of this HWNAS data set, architectures do not matter
        hw = hwnas_data[hwnas_data[hw_col0] == ds_metric]
        del hw[hw_col0]
        hw = hw.squeeze().to_numpy()
        hw_mean, hw_std = np.mean(hw), np.std(hw)
        hw -= hw_mean
        hw /= hw_std
        hw = np.sort(hw)

        # pair [accuracy, hw metric] in a 2D numpy array. both were sorted by arc names, so the order matches
        assert len(nb201_lookup[ds]) == num_arcs
        assert len(hw) == num_arcs
        paired_values = np.zeros(shape=(len(nb201_lookup[ds]), 2), dtype=np.float32)
        paired_values[:, 0] = nb201_lookup[ds]
        paired_values[:, 1] = hw

        #
        data = []
        for j, na in enumerate(num_architectures):
            for n in range(start_trial, end_trial):
                subset_indices = np.random.choice(arc_indices, size=na, replace=False)
                subset = paired_values[subset_indices, :]

                for acc_nv in acc_noise_values:
                    # currently the (0, 0) noise baseline is not computed...
                    # thus the first value is not always the highest if acc_nv>0
                    hvs = []
                    corrs = []
                    best_actual = []
                    best_supposed = []
                    best_actual_p = []
                    for hw_nv in noise_values:
                        res = get_noisy_best(subset, acc_noise=acc_nv, hw_noise=hw_nv)
                        hvs.append(res['hv'])
                        corrs.append(res['kt'])
                        best_actual.append(res['actually_best'])
                        best_supposed.append(res['supposedly_best'])
                        best_actual_p.append(res['actually_best_pareto'])

                    # measure acc/hw opportunity loss for each picked architecture, compared to true pareto front
                    best_actual[0] = best_actual[0][best_actual[0][:, 0].argsort()]
                    diff_acc_all, diff_acc_par = [0.0], [0.0]
                    diff_hw_all, diff_hw_par = [0.0], [0.0]
                    for i2 in range(len(best_actual)):
                        # the first one is always for std=0, can be skipped
                        if i2 > 0:
                            diff_acc_cur, diff_hw_cur = [], []
                            for i3, supp in enumerate(best_actual[i2]):
                                # find the true point that beats it
                                for best in best_actual[0]:
                                    if all(best <= supp):
                                        diff_acc_cur.append(abs(best[0] - supp[0]))
                                        break
                                for best in reversed(best_actual[0]):
                                    if all(best <= supp):
                                        diff_hw_cur.append(abs(best[1] - supp[1]))
                                        break
                            assert len(diff_acc_cur) == len(diff_hw_cur) == best_actual[i2].shape[0]
                            diff_acc_all.append(np.mean(diff_acc_cur))
                            diff_hw_all.append(np.mean(diff_hw_cur))
                    for i2 in range(len(best_actual_p)):
                        # the first one is always for std=0, can be skipped
                        if i2 > 0:
                            diff_acc_cur, diff_hw_cur = [], []
                            for i3, supp in enumerate(best_actual_p[i2]):
                                # find the true point that beats it
                                for best in best_actual[0]:
                                    if all(best <= supp):
                                        diff_acc_cur.append(abs(best[0] - supp[0]))
                                        break
                                for best in reversed(best_actual[0]):
                                    if all(best <= supp):
                                        diff_hw_cur.append(abs(best[1] - supp[1]))
                                        break
                            assert len(diff_acc_cur) == len(diff_hw_cur) == best_actual_p[i2].shape[0]
                            diff_acc_par.append(np.mean(diff_acc_cur))
                            diff_hw_par.append(np.mean(diff_hw_cur))

                    if do_plot:
                        # nice scatter plot
                        plt.close('all')
                        plt.rc('text.latex', preamble=r'\usepackage{textgreek}')
                        # best = [b[b[:, 0].argsort()] * (-1) for b in best]
                        best_actual = [b[b[:, 0].argsort()] for b in best_actual]
                        best_supposed = [b[b[:, 0].argsort()] for b in best_supposed]
                        best_actual_p = [b[b[:, 0].argsort()] for b in best_actual_p]

                        plt.scatter(subset[:, 1], -subset[:, 0], s=0.1, label="all architectures")
                        plt.step(best_actual[0][:, 1], -best_actual[0][:, 0], "r", where="pre", label="true pareto front")
                        plt.step(best_supposed[1][:, 1], -best_supposed[1][:, 0], "orange", where="pre", label="predicted pareto front")
                        plt.scatter(best_actual[1][:, 1], -best_actual[1][:, 0], c="orange", label="selected architectures", s=60)
                        plt.legend()
                        plt.ylabel("accuracy")
                        plt.xlabel("normalized %s" % ds_metric)
                        _, y2 = plt.ylim()
                        plt.ylim((y2/2, y2))
                        plt.gcf().set_size_inches((10, 5))
                        u.save_cur_plot("sim", "demo", "%s_%d_scatter" % (ds_metric, n))

                        # plot only pareto fronts
                        plt.close('all')
                        plt.step(best_actual[0][:, 1], -best_actual[0][:, 0], "o-", c="r", where="pre", label="true pareto front, HV=%.2f" % hvs[0])
                        plt.step(best_actual_p[1][:, 1], -best_actual_p[1][:, 0], "--", c="orange", where="pre",
                                 label="discovered pareto front, HV=%.2f" % hvs[1])
                        plt.scatter(best_actual[1][:, 1], -best_actual[1][:, 0], c="orange",
                                    label=r"selected arch., $MRA_{all}=%.2f$%s, $MRA_{pareto}=%.2f$%s" %
                                          (diff_acc_all[1]*100, '%', diff_acc_par[1]*100, '%'))
                        plt.legend(loc="lower right")
                        plt.ylabel("accuracy")
                        plt.xlabel("normalized %s" % ds_metric)
                        plt.ylim((y2/2, y2))
                        plt.grid()
                        plt.gcf().set_size_inches((5, 5))
                        u.save_cur_plot("sim", "demo", "%s_%d_front" % (ds_metric, n))

                    data.append((ds, metric, na, n, acc_nv, hvs, corrs, diff_acc_all, diff_acc_par, diff_hw_all, diff_hw_par))
                    del best_supposed, best_actual, best_actual_p

            print("completed %d/%d, %d/%d, t=%.1f" % (i+1, len(all_hw_datasets), j+1, len(num_architectures), time.time() - t1))

        # as dataframe, merge, save
        if not do_plot:
            df = pd.DataFrame(data, columns=columns)
            if isinstance(full_df, pd.DataFrame):
                full_df = full_df.append(df)
            else:
                full_df = df
            full_df.to_csv(hv_csv_path)
            print("updated dataframe after [%s]: %s" % (ds_metric, hv_csv_path))
    return full_df


def plot_dist(std=None):
    """
    plot randomly sampled values that approximate the common predictor errors
    """
    num_bins = 50
    for i, noise in enumerate([0.1 * v for v in range(11)]):
        if isinstance(std, float):
            noise = std
        y = get_noise(noise, 10000)
        plt.close('all')
        plt.figure(figsize=(4, 3.5))
        plt.hist(y, bins=50, density=True, label="mixed dist. generated with std=%.1f" % noise)

        # add pdfs
        x_min, x_max = plt.xlim()
        x = np.linspace(x_min, x_max, num_bins*5)
        for (name, fun) in [
            ("normal", stats.norm),
            # ("cauchy", stats.cauchy),
        ]:
            args = stats.norm.fit(y)
            plt.plot(x, fun.pdf(x, *args), label="%s fit, std=%.3f" % (name, args[1]))

        plt.legend()
        plt.xlabel("deviation of the simulated predictions")
        plt.ylabel("density")
        plt.title("Simulated predictor deviations")
        plt.tight_layout()
        u.save_cur_plot("sim", "approx", "%d" % i)
        # plt.show()


def plot_sim_data(hv_csv_path: str):
    # load data, get all keys
    df = pd.read_csv(hv_csv_path)
    df.drop(columns=(df.columns[0]))

    # figure out what's the column name for hypervolume values (it may change, depending on sampled std)
    hv_col = 0
    for i, col in enumerate(df.columns):
        if col.startswith('hv for'):
            hv_col = i
            break

    all_datasets = df['dataset'].unique()
    all_metrics = df['metric'].unique()
    all_numarcs = df['num_arc'].unique()
    all_accstds = df['acc std'].unique()
    all_hwstds = np.array(eval("[%s" % df.columns[hv_col].rsplit('[')[1]))
    df.rename(columns={df.columns[hv_col]: "hv"}, inplace=True)
    # cast some columns to numpy arrays
    for key, mul in [
        ("hv", False),
        ("kt", False),
        ("acc_all_to_par", True),
        ("acc_par_to_par", True),
        ("hw_all_to_par", False),
        ("hw_par_to_par", False)
    ]:
        df[key] = df[key].map(lambda x: np.array(eval(x)))
        if mul:
            df[key] = df[key].map(lambda x: x*100)
    df["hv_rem"] = df["hv"].map(lambda x: x / x[0])         # remaining of the hv

    # only for one now
    df0 = df[df["acc std"] == 0.0]

    """
    plot [std, err all/par]
    """

    gb = ['dataset', 'metric']
    df1 = df0.copy().set_index(gb).apply(list)

    for ds in itertools.chain(df1.index.unique(), [None]):
        if ds is not None:
            dfx = df1.loc[ds]
        else:
            dfx = df0

        plt.close('all')
        plt.figure(figsize=(5, 3.5))
        metrics = ['acc_all_to_par', 'acc_par_to_par']
        dfx = dfx[metrics + ['kt']]
        r_mean_ = dfx.apply(lambda x: np.mean(x, 0))

        plot_iter = iter(u.plot_markers_cols)
        for metric in metrics:
            marker, col = plot_iter.__next__()
            plt.plot(all_hwstds, r_mean_[metric], marker, label=u.metric_names[metric+"_np"], c=col)
            # plt.fill_between(all_hwstds, np.clip(r_mean_[metric]-r_std_[metric], 0, 100), np.clip(r_mean_[metric]-r_std_[metric], 0, 100), alpha=0.1, color=col)
        # finish plot
        plt.xlabel("%s / %s" % (u.metric_names['diff_std'], u.metric_names['kendalltau']))
        plt.ylabel(u.metric_names['mra'])
        plt.grid()
        plt.legend()
        # change xticks
        plt.subplots_adjust(bottom=0.17) # make room on bottom
        locs, _ = plt.xticks()
        locs = locs[1:-1]
        kt_mean = dfx['kt'].mean()
        assert len(locs) == 6, "hardcoded for convenience"
        assert len(kt_mean) == 11, "hardcoded for convenience"
        kt_mean = [kt_mean[0], kt_mean[2], kt_mean[4], kt_mean[6], kt_mean[8], kt_mean[10]]
        plt.xticks(locs, ["%.1f/%.2f" % (lc, kt) for lc, kt in zip(locs, kt_mean)])
        u.save_cur_plot("sim", "eval", "other", "mra-%s" % str(ds) if isinstance(ds, tuple) else "mean")

    """
    plot [std, metric] averaged over datasets/num arc
    """

    def plot_grouped(dfx_: pd.DataFrame, col_: str, title_: str, xtype: str, ylabel: str, plot_name_: str,
                     group_by: list, add_mean=True, subdirs=(), clip0=0.0, clip1=1.0):
        grouped_ = dfx_.set_index(group_by)[col_].groupby(group_by).apply(list)
        r_mean = grouped_.apply(lambda x: np.mean(x, 0))
        r_std = grouped_.apply(lambda x: np.std(x, 0))
        plt.close('all')
        plt.figure(figsize=(5, 3.5))
        # plt.rcParams.update({
        #     "text.usetex": True,
        #     "font.family": "sans-serif",
        #     "font.sans-serif": ["Helvetica"],
        # })
        plot_iter = iter(u.plot_markers_cols)
        # individually
        for i, (rmv, rsv) in enumerate(zip(r_mean, r_std)):
            marker, col = plot_iter.__next__()
            name = "-".join([str(s) for s in r_mean.index[i]]) if isinstance(r_mean.index[i], Iterable) else str(r_mean.index[i])
            plt.plot(all_hwstds, rmv, marker, label=name, c=col)
            plt.fill_between(all_hwstds, np.clip(rmv-rsv, clip0, clip1), np.clip(rmv+rsv, clip0, clip1), alpha=0.1, color=col)
        # average
        if add_mean:
            marker, col = plot_iter.__next__()
            mm = r_mean.mean()
            ms = r_std.mean()
            plt.plot(all_hwstds, mm, marker, label="mean", c=col)
            plt.fill_between(all_hwstds, np.clip(mm-ms, clip0, clip1), np.clip(mm+ms, clip0, clip1), alpha=0.2, color=col)
        # finish plot
        plt.ylabel(ylabel)
        plt.grid()
        plt.title(title_)

        plt.legend()

        # plt.legend(ncol=3)
        # plt.ylim((2, 4.5))

        # change xticks
        if xtype == "diff_std":
            plt.xlabel("%s / %s" % (u.metric_names['diff_std'], u.metric_names['kendalltau']))
            plt.subplots_adjust(bottom=0.17)  # make room on bottom
            locs, _ = plt.xticks()
            locs = locs[1:-1]
            kt_mean = dfx_['kt'].mean()
            assert len(locs) == 6, "hardcoded for convenience"
            assert len(kt_mean) == 11, "hardcoded for convenience"
            kt_mean = [kt_mean[0], kt_mean[2], kt_mean[4], kt_mean[6], kt_mean[8], kt_mean[10]]
            plt.xticks(locs, ["%.1f/%.2f" % (lc, kt) for lc, kt in zip(locs, kt_mean)])
        else:
            plt.xlabel(xtype)
        plt.tight_layout()
        u.save_cur_plot("sim", "eval", *subdirs, plot_name_)

    for (fn, metric, col, mean) in [
        ("err-aatp", "acc_all_to_par", None, True),
        ("err-aptp", "acc_par_to_par", None, True),
        ("kt", "kendalltau", "kt", True),
        ("hvrem", "hv_rem", None, False),
        ("hv", "hv", None, False),
    ]:
        clip0_, clip1_ = (0.0, 100.0) if 'err' in fn else (0.0, 1.0)

        for ds in itertools.chain(all_datasets, [None]):
            plot_name = "std-%s_%s" % (fn, ds if ds is not None else "ds-mean")
            if ds is not None:
                dfx = df0[df0['dataset'] == ds]
                title = ds
            else:
                dfx = df0
                title = "mean over all datasets"
            plot_grouped(dfx, col if col is not None else metric, title,
                         xtype='diff_std',
                         ylabel=u.metric_names[metric],
                         plot_name_=plot_name,
                         group_by=['num_arc'],
                         add_mean=mean,
                         subdirs=("by_ds",),
                         clip0=clip0_, clip1=clip1_)

        for na in itertools.chain(all_numarcs, [None]):
            plot_name = "std-%s_%s" % (fn, str(na) if na is not None else "arc-mean")
            if na is not None:
                dfx = df0[df0['num_arc'] == na]
                title = "%d architectures" % na
            else:
                dfx = df0
                title = "mean over any number of architectures"
            plot_grouped(dfx, col if col is not None else metric, title,
                         xtype='diff_std',
                         ylabel=u.metric_names[metric],
                         plot_name_=plot_name,
                         group_by=['dataset', 'metric'],
                         add_mean=True,
                         subdirs=("by_num_arc",),
                         clip0=clip0_, clip1=clip1_)


def plot_howto():
    """
    how to measure differences in accuracy/hw to the pareto front
    """
    plt.close('all')
    x, y = [17, 20, 23, 26, 30], [45, 50, 53, 56, 60]
    plt.step(x, y, "o--", c="r", where="post", label="true pareto front")
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi+0.5, "A%d" % (i+1))
    plt.scatter([28], [48], c="orange", label="selected architecture")
    plt.text(28, 47, "C1")

    plt.arrow(28, 48.5, 0, 7.2, width=0.05, length_includes_head=True, color="black")
    plt.text(28.2, 52, "accuracy")
    plt.text(28.2, 51, "difference")

    plt.arrow(27.7, 48, -7.4, 0, width=0.07, length_includes_head=True, color="black")
    plt.text(22, 47, "HW metric difference")

    plt.xlabel("hardware metric")
    plt.ylabel("accuracy [%]")
    plt.ylim((y[0]-1, y[-1]+2))
    plt.legend()
    u.save_cur_plot("sim", "example", "howto")


def plot_howto2():
    """
    how to measure hyper-volume
    """
    plt.close('all')
    x, y = [17, 20, 23, 26, 30], [15, 24, 32, 46, 51]
    x2, y2 = [], []
    for i in range(len(x)-1):
        x2.append(x[i])
        y2.append(y[i])
        x2.append(x[i+1]-0.01)
        y2.append(y[i])
    # last point
    x2.append(x[-1])
    y2.append(y[-1])
    x2.append(x[-1]*1.1)
    y2.append(y[-1])

    x2, y2 = np.array(x2), np.array(y2)
    plt.step(x, y, "o--", c="r", where="post", label="pareto front")
    plt.fill_between(x2, np.zeros_like(y2), y2, alpha=0.15, color="r", label="hypervolume")

    plt.scatter([33.0], [0.0], c="black", label="reference point")
    plt.text(29, 1, "reference point")

    plt.arrow(30.2, 49, 2.6, 0, width=0.3, head_length=0.5, length_includes_head=True, color="black")
    plt.text(31, 46, "+10%")
    plt.arrow(18, 14.2, 0, -13.5, width=0.1, head_length=1.5, length_includes_head=True, color="black")
    plt.text(18.3, 8, "to 0")

    plt.xlabel("hardware metric")
    plt.ylabel("accuracy [%]")
    plt.ylim((-3, y[-1]+2))
    plt.legend()
    u.save_cur_plot("sim", "example", "howto2")


if __name__ == '__main__':
    hv_csv_path_ = u.sim_dir + "/sim_%s.csv"

    # appendix plots, how MRA/hypervolume are measured
    # plot_howto()
    # plot_howto2()

    # example plots of the simulated deviation distribution
    # plot_dist(std=0.5)

    # simulating architecture selection, generating the results
    # create csvs of hypervolume samples one at a time so interruptions are possible
    # dfs = []
    # for i__ in range(1):
    #     print("-"*100)
    #     dfs.append(collect_sim_data(hv_csv_path_ % str(i__), start_trial=100 * i__, end_trial=100 * (i__ + 1)))
    # df = pd.concat(dfs, axis=0)
    # df = df.reset_index(drop=True)
    # df.to_csv(hv_csv_path_ % 'hwnas')

    # plotting the simulation results
    plot_sim_data(hv_csv_path_ % 'hwnas')
