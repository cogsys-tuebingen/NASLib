import random
import torch
import itertools

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core.graph import Graph


OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1"]


class HWNasSearchSpace(Graph):
    """
    Interface to the tabular benchmark of HW-NAS,
    for efficiency this Graph can not be run.
    """

    QUERYABLE = True

    _all_architectures = list(itertools.product([0, 1, 2, 3, 4], repeat=6))
    random.shuffle(_all_architectures)

    def __init__(self):
        super().__init__()
        self.num_classes = self.NUM_CLASSES if hasattr(self, "NUM_CLASSES") else 10
        self.op_indices = None

        self.max_epoch = 199
        self.space_name = "hwnas"

    def sample_random_architecture(self, dataset_api=None):
        """
        returns one of the possible architectures, avoids returning the same
        """
        op_indices = self._all_architectures.pop()
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
        Query results from hwnas
        """
        assert isinstance(dataset_api, dict)
        assert len(dataset_api["raw"]) == 15625

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
        self.get_op_indices()
        nbrs = []
        for edge in range(len(self.op_indices)):
            available = [o for o in range(len(OP_NAMES)) if o != self.op_indices[edge]]

            for op_index in available:
                nbr_op_indices = self.op_indices.copy()
                nbr_op_indices[edge] = op_index
                nbr = HWNasSearchSpace()
                nbr.set_op_indices(nbr_op_indices)
                nbr_model = torch.nn.Module()
                nbr_model.arch = nbr
                nbrs.append(nbr_model)

        random.shuffle(nbrs)
        return nbrs

    def get_type(self):
        return "hwnas"

    @classmethod
    def reset_architectures(cls):
        cls._all_architectures = list(itertools.product([0, 1, 2, 3, 4], repeat=6))
        random.shuffle(cls._all_architectures)
