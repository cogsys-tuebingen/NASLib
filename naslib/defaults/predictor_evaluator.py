import codecs
import time
import json
import logging
import os
import numpy as np
import copy
import torch
from scipy import stats
from sklearn import metrics
import math
import matplotlib.pyplot as plt

from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils import generate_kfold, cross_validation

logger = logging.getLogger(__name__)

# just for some plotting
save_plots_dir = None
plot_counter = 0
# save_plots_dir = "/data/workspace/naslib/hw_predictors/_plots/err/"
print_data_dict = False


class PredictorEvaluator(object):
    """
    This class will evaluate a chosen predictor based on
    correlation and rank correlation metrics, for the given
    initialization times and query times.
    """

    def __init__(self, predictor, config=None):

        self.predictor = predictor
        self.config = config
        self.experiment_type = config.experiment_type

        self.test_size = config.test_size
        self.train_size_single = config.train_size_single
        self.train_size_list = config.train_size_list
        self.fidelity_single = config.fidelity_single
        self.fidelity_list = config.fidelity_list
        self.max_hpo_time = config.max_hpo_time

        self.dataset = config.dataset
        self.metric = Metric.VAL_ACCURACY
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.results = [config]

        # mutation parameters
        self.uniform_random = config.uniform_random
        self.mutate_pool = 10
        self.num_arches_to_mutate = 5
        self.max_mutation_rate = 3

    def adapt_search_space(
        self, search_space, load_labeled, scope=None, dataset_api=None
    ):
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.predictor.set_ss_type(self.search_space.get_type())
        self.load_labeled = load_labeled
        self.dataset_api = dataset_api

        # nasbench101 does not have full learning curves or hyperparameters
        if self.search_space.get_type() == "nasbench101":
            self.full_lc = False
            self.hyperparameters = False
        elif self.search_space.get_type() == "nasbench201":
            self.full_lc = True
            self.hyperparameters = True
        elif self.search_space.get_type() == "darts":
            self.full_lc = True
            self.hyperparameters = True
        elif self.search_space.get_type() == "nlp":
            self.full_lc = True
            self.hyperparameters = True
        elif self.search_space.get_type() in ["hwnas", "transnas_inf"]:
            self.full_lc = False
            self.hyperparameters = True
        else:
            raise NotImplementedError(
                "This search space is not yet implemented in PredictorEvaluator."
            )

    def get_full_arch_info(self, arch):
        """
        Given an arch, return the accuracy, train_time,
        and also a dict of extra info if required by the predictor
        """
        info_dict = {}
        accuracy = arch.query(
            # substituting acc with hardware metrics
            metric=Metric.HW, dataset=self.dataset, dataset_api=self.dataset_api
        )
        train_time = arch.query(
            metric=Metric.TRAIN_TIME, dataset=self.dataset, dataset_api=self.dataset_api
        )
        data_reqs = self.predictor.get_data_reqs()
        if data_reqs["requires_partial_lc"]:
            # add partial learning curve if applicable
            assert self.full_lc, "This predictor requires learning curve info"
            if type(data_reqs["metric"]) is list:
                for metric_i in data_reqs["metric"]:
                    metric_lc = arch.query(
                        metric=metric_i,
                        full_lc=True,
                        dataset=self.dataset,
                        dataset_api=self.dataset_api,
                    )
                    info_dict[f"{metric_i.name}_lc"] = metric_lc

            else:
                lc = arch.query(
                    metric=data_reqs["metric"],
                    full_lc=True,
                    dataset=self.dataset,
                    dataset_api=self.dataset_api,
                )
                info_dict["lc"] = lc
            if data_reqs["requires_hyperparameters"]:
                assert (
                    self.hyperparameters
                ), "This predictor requires querying arch hyperparams"
                for hp in data_reqs["hyperparams"]:
                    info_dict[hp] = arch.query(
                        Metric.HP, dataset=self.dataset, dataset_api=self.dataset_api
                    )[hp]
        return accuracy, train_time, info_dict

    def load_dataset(self, load_labeled=False, data_size=10, arch_hash_map=None):
        """
        There are two ways to load an architecture.
        load_labeled=False: sample a random architecture from the search space.
        This works on NAS benchmarks where we can query any architecture (nasbench101/201/301)
        load_labeled=True: sample a random architecture from a set of evaluated architectures.
        When we only have data on a subset of the search space (e.g., the set of 5k DARTS
        architectures that have the full training info).

        After we load an architecture, query the final val accuracy.
        If the predictor requires extra info such as partial learning curve info, query that too.
        """
        xdata = []
        ydata = []
        info = []
        train_times = []
        arch_hash_map = arch_hash_map if isinstance(arch_hash_map, dict) else {}

        while len(xdata) < data_size:
            if not load_labeled:
                arch = self.search_space.clone()
                arch.sample_random_architecture(dataset_api=self.dataset_api)
            else:
                arch = self.search_space.clone()
                arch.load_labeled_architecture(dataset_api=self.dataset_api)

            arch_hash = arch.get_hash()
            if arch_hash in arch_hash_map:
                continue
            else:
                arch_hash_map[arch_hash] = True

            accuracy, train_time, info_dict = self.get_full_arch_info(arch)
            xdata.append(arch)
            ydata.append(accuracy)
            info.append(info_dict)
            train_times.append(train_time)

        return [xdata, ydata, info, train_times], arch_hash_map

    def load_mutated_test(self, data_size=10, arch_hash_map=None):
        """
        Load a test set not uniformly at random, but by picking some random
        architectures and then mutation the best ones. This better emulates
        distributions in local or mutation-based NAS algorithms.
        """
        assert (
            self.load_labeled == False
        ), "Mutation is only implemented for load_labeled = False"
        xdata = []
        ydata = []
        info = []
        train_times = []
        arch_hash_map = arch_hash_map if isinstance(arch_hash_map, dict) else {}

        # step 1: create a large pool of architectures
        while len(xdata) < self.mutate_pool:
            arch = self.search_space.clone()
            arch.sample_random_architecture(dataset_api=self.dataset_api)
            arch_hash = arch.get_hash()
            if arch_hash in arch_hash_map:
                continue
            else:
                arch_hash_map[arch_hash] = True
            accuracy, train_time, info_dict = self.get_full_arch_info(arch)
            xdata.append(arch)
            ydata.append(accuracy)
            info.append(info_dict)
            train_times.append(train_time)

        # step 2: prune the pool down to the top 5 architectures
        indices = np.flip(np.argsort(ydata))[: self.num_arches_to_mutate]
        xdata = [xdata[i] for i in indices]
        ydata = [ydata[i] for i in indices]
        info = [info[i] for i in indices]
        train_times = [train_times[i] for i in indices]

        # step 3: mutate the top architectures to generate the full list
        while len(xdata) < data_size:
            idx = np.random.choice(self.num_arches_to_mutate)
            arch = xdata[idx].clone()
            mutation_factor = np.random.choice(self.max_mutation_rate) + 1
            for i in range(mutation_factor):
                new_arch = self.search_space.clone()
                new_arch.mutate(arch, dataset_api=self.dataset_api)
                arch = new_arch

            arch_hash = arch.get_hash()
            if arch_hash in arch_hash_map:
                continue
            else:
                arch_hash_map[arch_hash] = True
            accuracy, train_time, info_dict = self.get_full_arch_info(arch)
            xdata.append(arch)
            ydata.append(accuracy)
            info.append(info_dict)
            train_times.append(train_time)

        return [xdata, ydata, info, train_times], arch_hash_map

    def load_mutated_train(self, data_size=10, arch_hash_map=None, test_data=[]):
        """
        Load a training set not uniformly at random, but by picking architectures
        from the test set and mutating the best ones. There is still no overlap
        between the training and test sets. This better emulates local or
        mutation-based NAS algorithms.
        """
        assert (
            self.load_labeled == False
        ), "Mutation is only implemented for load_labeled = False"
        xdata = []
        ydata = []
        info = []
        train_times = []
        arch_hash_map = arch_hash_map if isinstance(arch_hash_map, dict) else {}

        while len(xdata) < data_size:
            idx = np.random.choice(len(test_data[0]))
            parent = test_data[0][idx]
            arch = self.search_space.clone()
            arch.mutate(parent, dataset_api=self.dataset_api)
            arch_hash = arch.get_hash()
            if arch_hash in arch_hash_map:
                continue
            else:
                arch_hash_map[arch_hash] = True
            accuracy, train_time, info_dict = self.get_full_arch_info(arch)
            xdata.append(arch)
            ydata.append(accuracy)
            info.append(info_dict)
            train_times.append(train_time)

        return [xdata, ydata, info, train_times], arch_hash_map

    def single_evaluate(self, train_data, test_data, fidelity):
        """
        Evaluate the predictor for a single (train_data / fidelity) pair
        """
        xtrain, ytrain, train_info, train_times = train_data
        xtest, ytest, test_info, _ = test_data
        train_size = len(xtrain)

        data_reqs = self.predictor.get_data_reqs()

        logger.info("Fit the predictor")
        if data_reqs["requires_partial_lc"]:
            """
            todo: distinguish between predictors that need LC info
            at training time vs test time
            """
            train_info = copy.deepcopy(train_info)
            test_info = copy.deepcopy(test_info)
            for info_dict in train_info:
                lc_related_keys = [key for key in info_dict.keys() if "lc" in key]
                for lc_key in lc_related_keys:
                    info_dict[lc_key] = info_dict[lc_key][:fidelity]

            for info_dict in test_info:
                lc_related_keys = [key for key in info_dict.keys() if "lc" in key]
                for lc_key in lc_related_keys:
                    info_dict[lc_key] = info_dict[lc_key][:fidelity]

        self.predictor.reset_hyperparams()
        fit_time_start = time.time()
        cv_score = 0
        if (
            self.max_hpo_time > 0
            and len(xtrain) >= 10
            and self.predictor.get_hpo_wrapper()
        ):

            # run cross-validation (for model-based predictors)
            hyperparams, cv_score = self.run_hpo(
                xtrain,
                ytrain,
                train_info,
                start_time=fit_time_start,
                metric="kendalltau",
            )
            self.predictor.set_hyperparams(hyperparams)

        self.predictor.fit(xtrain, ytrain, train_info)
        hyperparams = self.predictor.get_hyperparams()

        fit_time_end = time.time()
        test_pred = self.predictor.query(xtest, test_info)
        query_time_end = time.time()

        # If the predictor is an ensemble, take the mean
        if len(test_pred.shape) > 1:
            test_pred = np.mean(test_pred, axis=0)

        logger.info("Compute evaluation metrics")
        results_dict = self.compare(ytest, test_pred)
        if print_data_dict:
            print({t.get_op_indices(): (r, p) for t, r, p in zip(test_data[0], test_data[1], test_pred)})
        results_dict["train_size"] = train_size
        results_dict["fidelity"] = fidelity
        results_dict["train_time"] = np.sum(train_times)
        results_dict["fit_time"] = fit_time_end - fit_time_start
        results_dict["query_time"] = (query_time_end - fit_time_end) / len(xtest)
        if hyperparams:
            for key in hyperparams:
                results_dict["hp_" + key] = hyperparams[key]
        results_dict["cv_score"] = cv_score
        # print abridged results on one line:
        logger.info(
            "train_size: {}, fidelity: {}, kendall tau {}".format(
                train_size, fidelity, np.round(results_dict["kendalltau"], 4)
            )
        )
        # print entire results dict:
        print_string = ""
        for key in results_dict:
            if type(results_dict[key]) not in [str, set, bool]:
                # todo: serialize other types
                try:
                    print_string += key + ": {}, ".format(np.round(results_dict[key], 4))
                except:
                    pass
        logger.info(print_string)
        self.results.append(results_dict)
        """
        Todo: query_time currently does not include the time taken to train a partial learning curve
        """

    def evaluate(self):

        self.predictor.pre_process()

        logger.info("Load the test set")
        if self.uniform_random:
            test_data, arch_hash_map = self.load_dataset(
                load_labeled=self.load_labeled, data_size=self.test_size
            )
        else:
            test_data, arch_hash_map = self.load_mutated_test(data_size=self.test_size)

        logger.info("Load the training set")
        max_train_size = self.train_size_single

        if self.experiment_type in ["vary_train_size", "vary_both"]:
            max_train_size = self.train_size_list[-1]

        if self.uniform_random:
            full_train_data, _ = self.load_dataset(
                load_labeled=self.load_labeled,
                data_size=max_train_size,
                arch_hash_map=arch_hash_map,
            )
        else:
            full_train_data, _ = self.load_mutated_train(
                data_size=max_train_size,
                arch_hash_map=arch_hash_map,
                test_data=test_data,
            )

        # if the predictor requires unlabeled data (e.g. SemiNAS), generate it:
        reqs = self.predictor.get_data_reqs()
        unlabeled_data = None
        if reqs["unlabeled"]:
            logger.info("Load unlabeled data")
            unlabeled_size = max_train_size * reqs["unlabeled_factor"]
            [unlabeled_data, _, _, _], _ = self.load_dataset(
                load_labeled=self.load_labeled,
                data_size=unlabeled_size,
                arch_hash_map=arch_hash_map,
            )

        # some of the predictors use a pre-computation step to save time in batch experiments:
        self.predictor.pre_compute(full_train_data[0], test_data[0], unlabeled_data)

        if self.experiment_type == "single":
            train_size = self.train_size_single
            fidelity = self.fidelity_single
            self.single_evaluate(full_train_data, test_data, fidelity=fidelity)

        elif self.experiment_type == "vary_train_size":
            fidelity = self.fidelity_single
            for train_size in self.train_size_list:
                train_data = [data[:train_size] for data in full_train_data]
                self.single_evaluate(train_data, test_data, fidelity=fidelity)

        elif self.experiment_type == "vary_fidelity":
            train_size = self.train_size_single
            for fidelity in self.fidelity_list:
                self.single_evaluate(full_train_data, test_data, fidelity=fidelity)

        elif self.experiment_type == "vary_both":
            for train_size in self.train_size_list:
                train_data = [data[:train_size] for data in full_train_data]

                for fidelity in self.fidelity_list:
                    self.single_evaluate(train_data, test_data, fidelity=fidelity)

        else:
            raise NotImplementedError()

        self._log_to_json()
        return self.results

    def compare(self, ytest, test_pred):
        ytest = np.array(ytest)
        test_pred = np.array(test_pred)
        METRICS = [
            "mae",
            "rmse",
            "pearson",
            "spearman",
            "kendalltau",
            "kt_2dec",
            "kt_1dec",
            "precision_10",
            "precision_20",
        ]
        metrics_dict = {}

        try:
            metrics_dict["mae"] = np.mean(abs(test_pred - ytest))
            metrics_dict["rmse"] = metrics.mean_squared_error(
                ytest, test_pred, squared=False
            )

            # std of errors
            diff = (test_pred - ytest)
            metrics_dict["diff_mean"] = np.mean(diff)
            metrics_dict["diff_std"] = np.std(diff)

            # how are the differences distributed? which density function fits best? based on 100 samples
            diff100 = diff[:100]
            for (name, fit_fun) in [
                ("norm", stats.norm),
                ("cauchy", stats.cauchy),
                ("lognorm", stats.lognorm),
                ("t", stats.t),
                ("uniform", stats.uniform),
            ]:
                args = fit_fun.fit(diff100)
                tt_stat, tt_p = stats.ttest_ind(diff100, fit_fun.pdf(diff100, *args))
                metrics_dict["test_tt:%s:stats" % name] = (tt_stat, tt_p)
                metrics_dict["test_tt:%s:args" % name] = args

            # correlations
            corr_p = np.abs(np.corrcoef(ytest, test_pred)[1, 0])
            corr_s = stats.spearmanr(ytest, test_pred)[0]
            corr_k = stats.kendalltau(ytest, test_pred)[0]
            metrics_dict["pearson"] = corr_p
            metrics_dict["spearman"] = corr_s
            metrics_dict["kendalltau"] = corr_k
            metrics_dict["kt_2dec"] = stats.kendalltau(
                ytest, np.round(test_pred, decimals=2)
            )[0]
            metrics_dict["kt_1dec"] = stats.kendalltau(
                ytest, np.round(test_pred, decimals=1)
            )[0]
            for k in [10, 20]:
                top_ytest = np.array(
                    [y > sorted(ytest)[max(-len(ytest), -k - 1)] for y in ytest]
                )
                top_test_pred = np.array(
                    [
                        y > sorted(test_pred)[max(-len(test_pred), -k - 1)]
                        for y in test_pred
                    ]
                )
                metrics_dict["precision_{}".format(k)] = (
                    sum(top_ytest & top_test_pred) / k
                )

            # optional plots
            global save_plots_dir, plot_counter
            if isinstance(save_plots_dir, str):
                # plotting ytest / test_pred
                plt.close('all')
                plt.figure(figsize=(4, 3.5))
                plt.scatter(ytest, test_pred, label="KT=%.2f, SCC=%.2f, PCC=%.2f" % (corr_k, corr_s, corr_p), s=15)
                min_, max_ = np.min(ytest), np.max(ytest)
                plt.plot([min_, max_], [min_, max_], "r-")
                plt.xlabel("true values")
                plt.ylabel("predicted values")
                plt.title("Predictions and targets")
                plt.legend()
                plt.tight_layout()
                plt.savefig('%s/%d_err_pred.pdf' % (save_plots_dir, plot_counter))

                # plotting differences as histogram
                plt.close('all')
                plt.figure(figsize=(4, 3.5))
                plt.hist(diff, bins=50, density=True, label="deviations")
                x_min, x_max = plt.xlim()
                x = np.linspace(x_min, x_max, 500)
                args = stats.norm.fit(diff)
                plt.plot(x, stats.norm.pdf(x, *args), label="normal fit, std=%.3f" % metrics_dict["diff_std"])
                plt.axvline(x=0, color='red', linestyle='-')
                plt.legend()
                plt.xlabel("deviation of the predictions")
                plt.ylabel("density")
                plt.title("Predictor deviations")
                plt.tight_layout()
                plt.savefig('%s/%d_err_dist.pdf' % (save_plots_dir, plot_counter))
                plot_counter += 1

        except:
            for metric in METRICS:
                metrics_dict[metric] = float("nan")
        if np.isnan(metrics_dict["pearson"]) or not np.isfinite(
            metrics_dict["pearson"]
        ):
            logger.info("Error when computing metrics. ytest and test_pred are:")
            logger.info(ytest)
            logger.info(test_pred)

        return metrics_dict

    def run_hpo(
        self,
        xtrain,
        ytrain,
        train_info,
        start_time,
        metric="kendalltau",
        max_iters=5000,
    ):
        logger.info(f"Starting cross validation")
        n_train = len(xtrain)
        split_indices = generate_kfold(n_train, 3)
        # todo: try to run this without copying the predictor
        predictor = copy.deepcopy(self.predictor)

        best_score = -1e6
        best_hyperparams = None

        t = 0
        while t < max_iters:
            t += 1
            hyperparams = predictor.set_random_hyperparams()
            cv_score = cross_validation(
                xtrain, ytrain, predictor, split_indices, metric
            )
            if np.isnan(cv_score) or cv_score < 0:
                # todo: this will not work for mae/rmse
                cv_score = 0

            if cv_score > best_score or t == 0:
                best_hyperparams = hyperparams
                best_score = cv_score
                logger.info(f"new best score={cv_score}, hparams = {hyperparams}")

            if (time.time() - start_time) > self.max_hpo_time * (
                len(xtrain) / 1000
            ) + 20:
                # we always give at least 20 seconds, and the time scales with train_size
                break

        if math.isnan(best_score):
            best_hyperparams = predictor.default_hyperparams

        logger.info(f"Finished {t} rounds")
        logger.info(f"Best hyperparams = {best_hyperparams} Score = {best_score}")
        self.predictor.hyperparams = best_hyperparams

        return best_hyperparams.copy(), best_score

    def _log_to_json(self):
        """log statistics to json file"""
        if not os.path.exists(self.config.save):
            os.makedirs(self.config.save)
        with codecs.open(
            os.path.join(self.config.save, "errors.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(self.results, file, separators=(",", ":"))
