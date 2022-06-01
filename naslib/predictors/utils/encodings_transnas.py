import numpy as np
import logging

logger = logging.getLogger(__name__)


OPS = ["", "normal", "channel*2", "resolution/2", "channel*2 and resolution/2", "-"]  # last should be unnecessary
NUM_OPS = len(OPS)

one_hot_transnas_inf = []
for i in range(NUM_OPS):
    o = [0]*NUM_OPS
    o[i] = 1
    one_hot_transnas_inf.append(o)


def encode_adjacency_one_hot(arch):
    encoding = arch.get_op_indices()
    one_hot = []
    for e in encoding:
        one_hot = [*one_hot, *one_hot_transnas_inf[e]]
    return one_hot


def encode_gcn_transnas_inf(arch):
    """
    Input:
    a list of categorical ops starting from 0
    """
    ops = arch.get_op_indices()
    # offset ops list by one, add input and output to ops list
    ops = [op + 1 for op in ops]
    ops = [0, *ops, 7]
    ops_onehot = np.array([[i == op for i in range(8)] for op in ops], dtype=np.float32)
    matrix = np.eye(8)
    # matrix = np.transpose(matrix)
    return {
        "num_vertices": len(ops),
        "adjacency": matrix,
        "operations": ops_onehot,
        "mask": np.ones(shape=(len(ops),), dtype=np.float32),
        "val_acc": 0.0,
    }


def encode_bonas_transnas_inf(arch):
    """
    Input:
    a list of categorical ops starting from 0
    """
    ops = arch.get_op_indices()
    # offset ops list by one, add input and output to ops list
    ops = [op + 1 for op in ops]
    ops = [0, *ops, 7]
    ops_onehot = np.array([[i == op for i in range(8)] for op in ops], dtype=np.float32)
    matrix = np.eye(8)

    matrix = add_global_node(matrix, True)
    ops_onehot = add_global_node(ops_onehot, False)

    matrix = np.array(matrix, dtype=np.float32)
    ops_onehot = np.array(ops_onehot, dtype=np.float32)

    dic = {"adjacency": matrix, "operations": ops_onehot, "val_acc": 0.0}
    return dic


def add_global_node(mx, ifAdj):
    """add a global node to operation or adjacency matrixs, fill diagonal for adj and transpose adjs"""
    if ifAdj:
        mx = np.column_stack((mx, np.ones(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        np.fill_diagonal(mx, 1)
        mx = mx.T
    else:
        mx = np.column_stack((mx, np.zeros(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        mx[mx.shape[0] - 1][mx.shape[1] - 1] = 1
    return mx


def encode_seminas_transnas_inf(arch):
    """
    Input:
    a list of categorical ops starting from 0
    """
    ops = arch.get_op_indices()
    # offset ops list by one, add input and output to ops list
    ops = [op + 1 for op in ops]
    ops = [0, *ops, 7]
    matrix = np.eye(8)
    dic = {
        "num_vertices": 8,
        "adjacency": matrix,
        "operations": ops,
        "mask": np.array([i < 8 for i in range(8)], dtype=np.float32),
        "val_acc": 0.0,
    }

    return dic


def encode_transnas_inf(arch, encoding_type="adjacency_one_hot"):
    assert encoding_type in ['adjacency_one_hot', 'path'], "Other encoding types make no sense for a sequential network"
    return encode_adjacency_one_hot(arch)

    if encoding_type == "adjacency_one_hot":
        return encode_adjacency_one_hot(arch)

    elif encoding_type == "path":
        # this network is purely sequential
        return encode_adjacency_one_hot(arch)

    elif encoding_type == "gcn":
        return encode_gcn_transnas_inf(arch)

    elif encoding_type == "bonas":
        return encode_bonas_transnas_inf(arch)

    elif encoding_type == "seminas":
        return encode_seminas_transnas_inf(arch)

    else:
        logger.info(
            "{} is not yet supported as a predictor encoding".format(encoding_type)
        )
        raise NotImplementedError()
