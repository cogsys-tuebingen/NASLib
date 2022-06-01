"""
collect errors.json in dir into a csv
"""


import json
from pathlib import Path
import pandas as pd
import _hw.utils as u


def collect_from_dir(dir_: str, save_dir: str):
    cfg_cols = ('dataset', 'predictor', 'seed')
    d_cols = ('train_size', 'pearson', 'spearman', 'kendalltau', 'train_time', 'fit_time', 'query_time', 'diff_mean', 'diff_std')
    d_p_values = ('tt:norm', 'tt:cauchy', 'tt:lognorm', 'tt:t', 'tt:uniform')
    data = []
    paths = list(Path(dir_).glob("**/errors.json"))

    for i, p in enumerate(paths):
        print("%d / %d" % (i+1, len(paths)))

        with open(p) as f:
            lst = json.load(f)
            cfg = lst.pop(0)

            for d in lst:
                t = []
                for c in cfg_cols:
                    t.append(cfg[c])
                for c in d_cols:
                    t.append(d[c])
                for c in d_p_values:
                    t.append(d['test_%s:stats' % c][1])
                data.append(t)

    save_path = "%s/all.csv" % save_dir
    df = pd.DataFrame(data, columns=cfg_cols + d_cols + d_p_values)
    df.to_csv(save_path)
    print("saved to: %s" % save_path)


def split_(save_dir: str):
    save_path = "%s/all.csv" % save_dir
    df = pd.read_csv(save_path)

    # remove old
    df = df[df['dataset'] != 'CPU_latency']

    # results on hard HW-NAS datasets
    df_hw = df[df['dataset'].isin(u.hard_datasets_hwnas)]
    df_hw = df_hw.drop(columns=[df_hw.columns[0]]).reset_index(drop=True)
    df_hw.to_csv('%s/results_hwnas.csv' % save_dir)

    # results on TransNAS datasets
    df_t = df[df['dataset'].isin(u.datasets_transnas)]
    df_t = df_t.drop(columns=[df_t.columns[0]]).reset_index(drop=True)
    df_t.to_csv('%s/results_transnas.csv' % save_dir)

    # simple predictors on all
    df_simple = df[df['predictor'].isin(['lin_reg', 'xgb'])]
    # dfs = df_simple[df_simple['dataset'].isin(u.datasets_transnas)]
    # dfs = dfs.drop(columns=[dfs.columns[0]]).reset_index(drop=True)
    # dfs.to_csv('%s/simple_transnas.csv' % save_dir)

    dfs = df_simple[~df_simple['dataset'].isin(u.datasets_transnas)]
    dfs = dfs.drop(columns=[dfs.columns[0]]).reset_index(drop=True)
    dfs.to_csv('%s/simple_hwnas.csv' % save_dir)


if __name__ == '__main__':
    read_path = u.results_dir
    save_dir_ = u.csv_dir
    collect_from_dir(read_path, save_dir_)
    split_(save_dir_)
