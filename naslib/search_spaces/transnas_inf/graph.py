import copy
import random
import torch
import itertools

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core.graph import Graph
from naslib.utils.get_dataset_api import get_dataset_api


OPS = ["", "normal", "channel*2", "resolution/2", "channel*2 and resolution/2", "-"]  # last should be unnecessary


class LazyTransNASLookupSearchSpace(Graph):
    """
    Interface to the tabular benchmark of TransNAS latency,
    for efficiency this Graph can not be run.
    """

    QUERYABLE = True

    _all_architectures = None
    _all_architectures_cur = None

    def __init__(self, csv_name="transnas_inf", csv_ds="class_scene"):
        super().__init__()
        self.num_classes = self.NUM_CLASSES if hasattr(self, "NUM_CLASSES") else -1
        self.op_indices = None
        ds = get_dataset_api(csv_name, csv_ds)['raw']
        self.__class__._all_architectures = list(ds.keys())
        self.reset_architectures()

        self.max_epoch = 1
        self.space_name = self.get_type()

    def sample_random_architecture(self, dataset_api=None):
        """
        returns one of the possible architectures, avoids returning the same
        """
        op_indices = self._all_architectures_cur.pop()
        self.set_op_indices(op_indices)

    def query(
        self,
        metric=None,
        dataset=None,
        path=None,
        epoch=-1,
        full_lc=False,
        dataset_api=None,
    ):
        """
        Query results
        """
        assert isinstance(dataset_api, dict)
        assert len(dataset_api["raw"]) == 3256

        assert isinstance(metric, Metric)

        if metric == Metric.HW:
            return dataset_api["normalized"][self.get_op_indices()]

        if metric in [Metric.TRAIN_TIME, Metric.TEST_TIME, Metric.VAL_TIME]:
            return dataset_api["raw"][self.get_op_indices()]

        return None

    def get_hash(self):
        return tuple(self.get_op_indices())

    def set_op_indices(self, op_indices):
        # This will update the edges in the naslib object to op_indices
        self.op_indices = op_indices

    def get_op_indices(self):
        if self.op_indices is None:
            raise NotImplementedError
        return self.op_indices

    def get_nbhd(self, dataset_api=None):
        # return all neighbors of the architecture
        raise NotImplementedError

    def get_type(self):
        return "transnas_inf"

    @classmethod
    def reset_architectures(cls):
        cls._all_architectures_cur = copy.deepcopy(cls._all_architectures)
        random.shuffle(cls._all_architectures_cur)
