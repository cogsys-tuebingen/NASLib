"""
compute the time spent
"""

import _hw.utils as u


CPU_MULTIPLIER = 2/24  # each run used 2/24 cpu cores


if __name__ == '__main__':
    for name in ["results_transnas", "results_hwnas"]:
        df = u.get_result_data(name)
        train_time = df["train_time"].sum()
        fit_time = df["fit_time"].sum()
        query_time = df["query_time"].sum()

        # print(name, train_time, fit_time, query_time)
        print(name, (fit_time + query_time) * CPU_MULTIPLIER / 60 / 60 / 24)  # seconds to days
