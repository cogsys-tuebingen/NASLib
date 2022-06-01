"""
make a [dataset, train_size] table of some value, sort by worst
print a [device, metric] table to compare lookup tables with another predictor
"""


from collections import defaultdict
import pandas as pd
import _hw.fit_lookup_table as flt
import _hw.utils as u
pd.options.display.float_format = '{:.3f}'.format


def print_table(df: pd.DataFrame, predictor="lin_reg", predictor2="xgb", metric="kendalltau"):
    df1 = df[df["predictor"] == predictor]
    df2 = df[df["predictor"] == predictor2]

    # only required stats
    of_interest = ["dataset", "seed", "train_size", metric]
    df1 = df1[of_interest]

    # mean data
    data_mean = {}
    sizes = df1["train_size"].unique()

    datasets = df1["dataset"].unique()
    for ds in datasets:
        df12 = df1[df1["dataset"] == ds]
        df_mean = df12.groupby(by="train_size").mean().reset_index()
        data_mean[ds] = df_mean[metric]

        df22 = df2[df2["dataset"] == ds]
        df_mean = df22.groupby(by="train_size").mean().reset_index()
        data_mean[ds] = data_mean[ds].append(df_mean[metric].tail(1))

    df_mean = pd.DataFrame(data_mean).T
    sizes_ext = [s for s in sizes]
    sizes_ext.append(predictor2)
    df_mean.columns = sizes_ext
    df_mean = df_mean.sort_values(sizes_ext[-2], ascending=True)
    print(df_mean.to_latex())


def print_lut_table(df: pd.DataFrame, predictor="lin_reg", size=124):
    """ table of lookup table results by device/dataset """
    # only required stats
    df1 = df[df["predictor"] == predictor]
    df1 = df1[df1["train_size"] == size]
    of_interest = ["dataset", "seed", "kendalltau"]
    df1 = df1[of_interest]

    all_metrics, all_devices = set(), set()
    lut_results = defaultdict(lambda: defaultdict(list))
    predictor_results = defaultdict(lambda: defaultdict(list))

    for ds in df1["dataset"].unique():
        dataset_device, cur_metric = ds.split('_', 1)
        dataset, device = dataset_device.rsplit('-', 1)
        all_metrics.add(cur_metric)
        all_devices.add(device)
        lut_results[cur_metric][device].append(flt.get_metric("hwnas", ds, "kendalltau"))
        predictor_results[cur_metric][device].append(df1[df1["dataset"] == ds]["kendalltau"].mean())

    # print table
    # headline
    print("", end="")
    for device in all_devices:
        print(" & %s" % device, end="")
    print("\\\\")
    # rows
    for metric in all_metrics:
        print(metric, end="")
        for device in all_devices:
            r = lut_results.get(metric, {}).get(device, [])
            if len(r) == 0:
                print(" &", end="")
            else:
                r = sum(r) / len(r)
                r2 = predictor_results.get(metric, {}).get(device, [])
                r2 = sum(r2) / len(r2)
                print(" & %.2f (%.2f)" % (r, r2), end="")
        print("\\\\")


if __name__ == '__main__':
    print_lut_table(u.get_result_data("simple_hwnas"), predictor="lin_reg")
    # print_table(u.get_result_data("simple_hwnas"), predictor="lin_reg", predictor2="xgb")
    # print_table(u.get_result_data("results_transnas"), predictor="lin_reg", predictor2="xgb")
