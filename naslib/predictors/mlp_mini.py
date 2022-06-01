import numpy as np
from naslib.predictors.mlp import MLPPredictor


class MiniMLPPredictor(MLPPredictor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.default_hyperparams = {
            "num_layers": 3,
            "layer_width": 32,
            "activation": "relu",
            "batch_size": 32,
            "lr": 0.001,
            "regularization": 0.2,
        }

    def set_random_hyperparams(self):

        if self.hyperparams is None:
            params = self.default_hyperparams.copy()

        else:
            params = {
                "num_layers": int(np.random.choice(range(2, 5))),
                "layer_width": int(np.random.choice([16, 32, 64, 128])),
                "activation": np.random.choice(["relu", "tanh", "hardswish"]),
                "batch_size": 32,
                "lr": np.random.choice([0.1, 0.01, 0.005, 0.001, 0.0001]),
                "regularization": 0.2,
            }

        self.hyperparams = params
        return params
