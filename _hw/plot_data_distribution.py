"""
plot how the datasets are distributed
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from _hw.utils import get_hwnas_data, get_transnas_data, save_cur_plot


def plot_distribution(dataframe: pd.DataFrame, normalize=False):
    hw_col0 = dataframe.columns[0]
    all_hw_datasets = dataframe[hw_col0].unique()

    for dataset in all_hw_datasets:
        df = dataframe[dataframe[hw_col0] == dataset]
        data = df.to_dict()
        del data[hw_col0]

        values = []
        for k, v in data.items():
            assert len(v) == 1
            for v2 in v.values():
                values.append(v2)
                break
        values = np.array(values)

        if normalize:
            mean, std = np.mean(values), np.std(values)
            values = (values - mean) / std

        plt.close('all')
        plt.title(dataset)
        plt.hist(values, 50, density=False, alpha=0.8)
        plt.xlabel("measurements (normalized)" if normalize else "measurements")
        plt.ylabel("occurrences")
        save_cur_plot("data_distribution", "hist_%s" % dataset)
        # plt.show()


if __name__ == '__main__':
    plot_distribution(get_hwnas_data())
    plot_distribution(get_transnas_data())
