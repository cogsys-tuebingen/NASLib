"""
creating a lookup table model from a dataset
basic approach:
- use a small number of baseline architectures, e.g. (0, 0, 0, 0, 0, 0)
- vary one operation at a time, measure differences
"""

import numpy as np
from scipy.stats import kendalltau
import _hw.utils as u
from naslib.utils import get_dataset_api


def fit(space: str, dataset: str, print_count=False) -> (np.array, np.array):
    """
    create/fit a tabular lookup model from the data,
    return true values and predicted values for all architectures (same order)
    """
    data = get_dataset_api(space, dataset=dataset)['normalized']

    # figure out max index at each position
    max_, min_ = [0 for _ in iter(data.keys()).__next__()], [10 for _ in iter(data.keys()).__next__()]
    for key in data.keys():
        for i, v in enumerate(key):
            if max_[i] < v:
                max_[i] = v
            if min_[i] > v:
                min_[i] = v

    # easy differences, hwnas
    increases, base = [[None]*(s+1) for s in max_], 0
    if len(data) == np.prod([s+1 for s in max_]):
        def data_lookup(idx_=0, value_=0) -> float:
            key_ = [0]*len(max_)
            key_[idx_] = value_
            return data[tuple(key_)]

        # measure increases over the [0, 0, ...0] time
        base = data_lookup(0, 0)
        for i, size in enumerate(max_):
            for j in range(size+1):
                increases[i][j] = (data_lookup(i, j) - base)

    # less easy differences, transnas
    elif space == "transnas_inf":
        # two base architectures with lots of neighbors
        arc1 = (1, 1, 1, 1, 4, 0)  # permute all except the 4
        arc2 = (1, 1, 1, 4, 0, 0)  # permute the remaining
        base, base2 = data[arc1], data[arc2]

        def data_lookup(arc_: tuple, idx_=0, value_=0):
            key_ = [a for a in arc_]
            key_[idx_] = value_
            return data[tuple(key_)]

        # arc1
        for idx in [0, 1, 2, 3, 5]:
            for v in range(min_[idx], max_[idx]+1):
                increases[idx][v] = (data_lookup(arc1, idx, v) - base)
        # arc2
        for idx in [4]:
            for v in range(min_[idx], max_[idx]+1):
                increases[idx][v] = (data_lookup(arc2, idx, v) - base2)

    else:
        raise NotImplementedError

    def get_prediction(t: tuple) -> float:
        return base + sum([increases[i_][tx] for i_, tx in enumerate(t)])

    if print_count:
        count = sum([sum([1 for inc in op_increases if inc is not None]) for op_increases in increases]) - len(increases) + 1
        print("fit a model for %s, it requires %d measurements" % (space, count))

    # measure / predict
    true_values, predicted_values = [], []
    for k, v in data.items():
        true_values.append(v)
        predicted_values.append(get_prediction(k))

    return np.array(true_values), np.array(predicted_values)


def get_metric(space: str, ds: str, metric: str):
    if metric in ['kendalltau']:
        true, pred = None, None
        if 'hwnas' in space:
            true, pred = fit('hwnas', ds)
        elif 'transnas' in space:
            true, pred = fit('transnas_inf', ds)

        return kendalltau(true, pred)[0]
    return None


if __name__ == '__main__':
    # fit("hwnas", u.hard_datasets_hwnas[0], print_count=True)
    # fit("transnas_inf", u.datasets_transnas[0], print_count=True)

    # fit the tabular lookup model to all datasets
    for ds in u.get_all_transnas_datasets():
        try:
            print("{ds:<40} {r:.3f}".format(ds=ds, r=get_metric("transnas_inf", ds, metric="kendalltau")))
        except AssertionError:
            pass
    for ds in u.get_all_hwnas_datasets():
        try:
            print("{ds:<40} {r:.3f}".format(ds=ds, r=get_metric("hwnas", ds, metric="kendalltau")))
        except AssertionError:
            pass
