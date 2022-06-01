import os
import pickle
import pandas as pd
import numpy as np

from naslib.utils.utils import get_project_root

"""
This file loads any dataset files or api's needed by the Trainer or PredictorEvaluator object.
They must be loaded outside of the search space object, because search spaces are copied many times
throughout the discrete NAS algos, which would lead to memory errors.
"""


def get_nasbench101_api(dataset=None):
    # load nasbench101
    from nasbench import api

    nb101_data = api.NASBench(
        os.path.join(get_project_root(), "data", "nasbench_only108.tfrecord")
    )
    return {"api": api, "nb101_data": nb101_data}


def get_nasbench201_api(dataset=None):
    """
    Load the NAS-Bench-201 data
    """
    if dataset == "cifar10":
        with open(
            os.path.join(
                get_project_root(), "data", "nb201_cifar10_full_training.pickle"
            ),
            "rb",
        ) as f:
            data = pickle.load(f)

    elif dataset == "cifar100":
        with open(
            os.path.join(
                get_project_root(), "data", "nb201_cifar100_full_training.pickle"
            ),
            "rb",
        ) as f:
            data = pickle.load(f)

    elif dataset == "ImageNet16-120":
        with open(
            os.path.join(
                get_project_root(), "data", "nb201_ImageNet16_full_training.pickle"
            ),
            "rb",
        ) as f:
            data = pickle.load(f)

    return {"nb201_data": data}


def get_csv_api(dataset=None, normalize=True, file="hwnas.csv"):
    """
    Load the data from a csv lookup (only contains architectures and hardware metric data)
    normalize to have comparable predictor mean/std statistics, across datasets
    """
    df = pd.read_csv(os.path.join(get_project_root(), "data", file))
    df = df[df[df.columns[0]] == dataset]
    data = df.to_dict()
    del data[df.columns[0]]
    arcs = []
    values = []
    for k, v in data.items():
        assert len(v) == 1
        for v2 in v.values():
            assert v2 > 0, "the %s dataset is incomplete, contains negative/nan values" % dataset
            arcs.append(eval(k))
            values.append(v2)
            break
    return_dict = {"raw": {k: v for k, v in zip(arcs, values)}}
    if normalize:
        mean, std = np.mean(values), np.std(values)
        values = (np.array(values) - mean) / std
    return_dict["normalized"] = {k: v for k, v in zip(arcs, values)}
    return return_dict


def get_darts_api(dataset=None, 
                  nb301_model_path='~/nb_models/xgb_v1.0', 
                  nb301_runtime_path='~/nb_models/lgb_runtime_v1.0'):
    # Load the nb301 training data (which contains full learning curves)
    
    data_path = os.path.join(get_project_root(), "data/nb301_full_training.pickle")
    assert os.path.isfile(data_path), "Download nb301_full_training.pickle from\
    https://drive.google.com/drive/folders/1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa?usp=sharing"
    with open(data_path, "rb") as f:
        nb301_data = pickle.load(f)
        nb301_arches = list(nb301_data.keys())

    # Load the nb301 performance and runtime models
    nb301_model_path = os.path.expanduser(nb301_model_path)
    nb301_runtime_path = os.path.expanduser(nb301_runtime_path)
    assert os.path.exists(nb301_model_path), "Download v1.0 models from\
    https://github.com/automl/nasbench301"
    assert os.path.exists(nb301_runtime_path), "Download v1.0 models from\
    https://github.com/automl/nasbench301"

    import nasbench301
    performance_model = nasbench301.load_ensemble(
        nb301_model_path
    )
    runtime_model = nasbench301.load_ensemble(
        nb301_runtime_path
    )
    nb301_model = [performance_model, runtime_model]
    return {
        "nb301_data": nb301_data,
        "nb301_arches": nb301_arches,
        "nb301_model": nb301_model,
    }


def get_nlp_api(dataset=None, 
                nlp_model_path='~/nbnlp_v01'):
    # Load the NAS-Bench-NLP data
    with open(os.path.join(get_project_root(), "data", "nb_nlp.pickle"), "rb") as f:
        nlp_data = pickle.load(f)
    nlp_arches = list(nlp_data.keys())
    
    # Load the NAS-Bench-NLP11 performance model
    import nasbench301
    performance_model = nasbench301.load_ensemble(
        os.path.expanduser(nlp_model_path)
    )

    return {
        "nlp_data": nlp_data, 
        "nlp_arches": nlp_arches, 
        "nlp_model":performance_model}


def get_dataset_api(search_space=None, dataset=None):

    if search_space == "hwnas":
        return get_csv_api(dataset=dataset, file="hwnas.csv")

    elif search_space == "transnas_inf":
        return get_csv_api(dataset=dataset, file="transnas_inf.csv")

    elif search_space == "nasbench101":
        return get_nasbench101_api(dataset=dataset)

    elif search_space == "nasbench201":
        return get_nasbench201_api(dataset=dataset)

    elif search_space == "darts":
        return get_darts_api(dataset=dataset)

    elif search_space == "nlp":
        return get_nlp_api(dataset=dataset)

    elif search_space == "test":
        return None

    else:
        raise NotImplementedError()
