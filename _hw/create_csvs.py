"""
creating data frames from HW-NAS, NAS-Bench 201 and TransNAS data, saving as csv for simple lookup
"""


import os
import pandas as pd
from collections import defaultdict
from hw_nas_bench_api import HWNASBenchAPI  # needs to be in the python path, there is no pip wheel?
from api import TransNASBenchAPI            # see https://www.noahlab.com.hk/opensource/vega/page/doc.html?path=datasets/transnasbench101
from nats_bench import create
import _hw.utils as u


_arc_str_to_idx = {k: i for i, k in enumerate(['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3'])}


def arc_str_to_idx(arc: str) -> tuple:
    ops_str = arc[1:-1].replace('|+', '').split('|')
    ops_str = [s.split('~')[0] for s in ops_str]
    return tuple([int(_arc_str_to_idx.get(s)) for s in ops_str])


def create_nasbench201(nats_bench_path: str, save_dir: str):
    assert os.path.isdir(nats_bench_path), "Is not a dir: %s" % nats_bench_path
    nats_data_sets = ['cifar10', 'cifar100', 'ImageNet16-120']  # ignore c10-valid
    hp = '200'
    api = create(nats_bench_path, 'tss', fast_mode=True, verbose=False)

    df_data_nb201 = defaultdict(dict)

    for i, arch_str in enumerate(api.meta_archs):
        print("nb201 %d/15625" % i)
        as_idx = arc_str_to_idx(arch_str)
        for ds in nats_data_sets:
            info_res = api.get_more_info(i, ds, iepoch=None, hp=hp, is_random=False)
            df_data_nb201[str(as_idx)][ds] = info_res['test-accuracy']

        api.arch2infos_dict.clear()

    # save dataframes
    df_hw = pd.DataFrame(df_data_nb201)
    path_save = "%s/nb201.csv" % save_dir
    df_hw.to_csv(path_save)
    print("saved nb201 csv: %s" % path_save)


def create_hwnas(hw_bench_path: str, save_dir: str):
    assert os.path.isfile(hw_bench_path), "Is not a file: %s" % hw_bench_path
    hw_api = HWNASBenchAPI(hw_bench_path, search_space="nasbench201")

    data_names = list(hw_api.HW_metrics[hw_api.search_space].keys())
    metrics = list(hw_api.HW_metrics[hw_api.search_space][data_names[0]].keys())
    metrics.remove("config")

    df_data_hw = defaultdict(dict)

    for data_name in data_names:
        architectures = hw_api.HW_metrics[hw_api.search_space][data_name]["config"]
        arcs_as_idx = [arc_str_to_idx(v['arch_str']) for v in architectures]

        for metric in metrics:
            metric_values = hw_api.HW_metrics[hw_api.search_space][data_name][metric]
            data = {k: v for k, v in zip(arcs_as_idx, metric_values)}
            assert len(data) == len(arcs_as_idx), "Redundant architectures?"

            # add to csv
            key = "%s-%s" % (data_name, metric)
            for k, v in data.items():
                df_data_hw[str(k)][key] = v

    # save dataframes
    df_hw = pd.DataFrame(df_data_hw)
    path_save = "%s/hwnas.csv" % save_dir
    df_hw.to_csv(path_save)
    print("saved hw csv: %s" % path_save)


def create_transnas(trans_bench_path: str, save_dir: str):
    api = TransNASBenchAPI(trans_bench_path)
    tasks = {t: 'test_loss' if 'test_loss' in api.metrics_dict[t] else 'test_l1_loss' for t in api.task_list}
    tasks_keys = list(tasks.keys())
    # 'test_l1_loss'
    data_test_losses = {}
    data_inference_times = {}
    for arc in api.all_arch_dict['macro']:
        arc_name = arc.split('-')[1]
        arc_name = arc_name + '0'*(6-len(arc_name))
        arc_name = '(%s)' % ', '.join(arc_name)
        # print(arc_name)

        m_loss, m_inf = [], []
        for t_k, t_n in tasks.items():
            m_loss.append(api.get_single_metric(arc, t_k, t_n, mode='best'))
            m_inf.append(api.get_model_info(arc, t_k, 'inference_time'))
        data_test_losses[arc_name] = m_loss
        data_inference_times[arc_name] = m_inf

    df = pd.DataFrame(data_inference_times, index=tasks_keys)
    path_save = "%s/transnas_inf.csv" % save_dir
    df.to_csv(path_save)
    print("saved transnas inference csv: %s" % path_save)

    df = pd.DataFrame(data_test_losses, index=tasks_keys)
    path_save = "%s/transnas_loss.csv" % save_dir
    df.to_csv(path_save)
    print("saved transnas loss csv: %s" % path_save)


if __name__ == '__main__':
    os.makedirs(u.data_dir, exist_ok=True)
    create_nasbench201(nats_bench_path="/data/datasets/NATS-tss-v1_0-3ffb9-simple/", save_dir=u.data_dir)
    create_hwnas(hw_bench_path="/data/datasets/bench/HW-NAS-Bench-v1_0.pickle", save_dir=u.data_dir)
    create_transnas(trans_bench_path="/data/datasets/transnas-bench/api_home/transnas-bench_v10141024.pth", save_dir=u.data_dir)
