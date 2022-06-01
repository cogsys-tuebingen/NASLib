import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from naslib.utils.utils import get_project_root


# paths
results_dir = "/tmp//naslib/results/"
plot_dir = "/tmp//naslib/plots/"
csv_dir = os.path.join(get_project_root(), "data")
sim_dir = os.path.join(get_project_root(), "data")
data_dir = os.path.join(get_project_root(), "data")


hard_datasets_hwnas = [
    'ImageNet16-120-raspi4_latency',
    'cifar100-pixel3_latency',

    # technically harder to linear regression to fit, but similar to the others
    # 'cifar10-pixel3_latency',
    # 'cifar10-raspi4_latency',
    # 'cifar100-raspi4_latency',

    # easier to fit, but different
    'cifar10-edgegpu_latency',
    'cifar100-edgegpu_energy',
    'ImageNet16-120-eyeriss_arithmetic_intensity',
]
datasets_transnas = [
    "class_scene",
    "class_object",
    "room_layout",
    "jigsaw",
    "segmentsemantic",
]


plot_markers_cols = [
    ("o--", "red"),
    ("v--", "green"),
    ("^--", "blue"),
    ("<--", "orange"),
    (">--", "purple"),
    ("X--", "cyan"),
    ("*--", "gold"),
    ("d--", "deepskyblue")
]


class PlotInfo:
    def __init__(self, name: str, marker: str, line: str, col: str):
        self.name = name
        self.marker = marker
        self.line = line
        self.col = col


plot_infos = {
    "lin_reg": PlotInfo("Lin. Reg.", "o--", "-", "green"),
    "bayes_lin_reg": PlotInfo("Bayes. Lin. Reg.", "X--", "--", "lime"),
    "ridge_reg": PlotInfo("Ridge Reg.", "*--", ":", "mediumspringgreen"),

    "xgb": PlotInfo("XGBoost", "v--", "-.", "red"),
    "ngb": PlotInfo("NGBoost", "<--", ":", "orange"),
    "lgb": PlotInfo("LGBoost", ">--", "-", "sandybrown"),
    "rf": PlotInfo("Random Forests", "^--", "--", "chocolate"),

    "sparse_gp": PlotInfo("Sparse GP", "D--", "-.", "gold"),
    "gp": PlotInfo("GP", "d--", ":", "khaki"),

    "bohamiann": PlotInfo("BOHAMIANN", "x--", "-", "lightgray"),

    "svmr": PlotInfo("SVM Reg.", "*--", "-", "gray"),

    "nao": PlotInfo("NAO", "v--", "--", "deepskyblue"),
    "gcn": PlotInfo("GCN", "<--", "-.", "blue"),
    "bonas": PlotInfo("BONAS", ">--", ":", "purple"),
    "bananas": PlotInfo("BANANAS", "^--", "-", "cyan"),

    "mlp": PlotInfo("MLP (large)", "*--", "--", "fuchsia"),
    "minimlp": PlotInfo("MLP (small)", "x--", "-.", "orchid"),

    "tabular": PlotInfo("Lookup Table", "-", "-", "black"),
}
df_names = {
    "results_transnas": "TransNAS",
    "results_hwnas": "HW-NAS",
}
metric_names = {
    "kendalltau": "Kendall's Tau",
    "diff_std": "Std. of prediction deviations",
    "train_time": "Time to train the predictor",
    "fit_time": "Time to fit the predictor",
    "query_time": "Time to query results",

    "hv": "hypervolume",
    "hv_rem": "retained hypervolume",
    "mra": "MRA [%]",
    "acc_all_to_par": "MRA all [%]",
    "acc_all_to_par_np": "all selected architectures",
    "acc_par_to_par": "MRA pareto [%]",
    "acc_par_to_par_np": "pareto-set of the selected architectures",
}
diff_predictors = ["lin_reg", "bayes_lin_reg", "sparse_gp", "gp", "nao", "gcn", "bonas", "bananas", "mlp", "minimlp", "ridge_reg"]
not_diff_predictors = ["xgb", "ngb", "lgb", "rf", "bohamiann", "svmr"]


def is_differentiable(predictor: str) -> bool:
    if predictor in diff_predictors:
        return True
    if predictor in not_diff_predictors:
        return False
    raise NotImplementedError(predictor)


def get_transnas_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(get_project_root(), "data", "transnas_inf.csv"))


def get_hwnas_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(get_project_root(), "data", "hwnas.csv"))


def get_nb201_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(get_project_root(), "data", "nb201.csv"))


def get_result_data(name: str) -> pd.DataFrame:
    return pd.read_csv("%s/%s.csv" % (csv_dir, name))


def get_all_transnas_datasets() -> [str]:
    df = get_transnas_data()
    return [s for s in df[df.columns[0]].unique()]


def get_all_hwnas_datasets() -> [str]:
    df = get_hwnas_data()
    return [s for s in df[df.columns[0]].unique()]


def save_cur_plot(*names):
    """ save current plot in dir, add subdirs by giving names """
    for k, v in {
        " ": "_",
        "/": "_"
    }.items():
        names = [n.replace(k, v) for n in names]
    path = plot_dir + "/".join(names) + ".pdf"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    print("saved %s" % path)


if __name__ == '__main__':
    print(get_all_hwnas_datasets())
