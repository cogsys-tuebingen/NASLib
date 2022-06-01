import numpy as np

from naslib.predictors.utils.encodings import encode
from naslib.predictors.predictor import Predictor
from sklearn.linear_model import LinearRegression as LR, Ridge, Lasso


class LinearRegressionBase(Predictor):
    _model_cls = LR
    _may_hpo = True

    def __init__(
        self,
        encoding_type="adjacency_one_hot",
        ss_type="nasbench201",
        zc=False,
        hpo_wrapper=False,
    ):
        super(Predictor, self).__init__()
        self.encoding_type = encoding_type
        self.ss_type = ss_type
        self.zc = zc
        self.hyperparams = None
        assert (not hpo_wrapper) or self._may_hpo, "%s has only one set of possible parameters" % self.__class__.__name__
        self.hpo_wrapper = hpo_wrapper

    @property
    def default_hyperparams(self):
        raise NotImplementedError

    def get_dataset(self, encodings, labels=None):
        if labels is None:
            return encodings
        else:
            return (encodings, (labels - self.mean) / self.std)

    def train(self, train_data):
        X_train, y_train = train_data
        model = self._model_cls(**self.hyperparams)
        return model.fit(X_train, y_train)

    def predict(self, data, **kwargs):
        return self.model.predict(data, **kwargs)

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()

        # normalize accuracies
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)

        if type(xtrain) is list:
            # when used in itself, we use
            xtrain = np.array(
                [
                    encode(arch, encoding_type=self.encoding_type, ss_type=self.ss_type)
                    for arch in xtrain
                ]
            )

            if self.zc:
                mean, std = -10000000.0, 150000000.0
                xtrain = [
                    [*x, (train_info[i] - mean) / std] for i, x in enumerate(xtrain)
                ]
            xtrain = np.array(xtrain)
            ytrain = np.array(ytrain)

        else:
            # when used in aug_lcsvr we feed in ndarray directly
            xtrain = xtrain
            ytrain = ytrain

        # convert to the right representation
        train_data = self.get_dataset(xtrain, ytrain)

        # fit to the training data
        self.model = self.train(train_data)

        # predict
        train_pred = np.squeeze(self.predict(xtrain))
        train_error = np.mean(abs(train_pred - ytrain))

        return train_error

    def query(self, xtest, info=None):

        if type(xtest) is list:
            #  when used in itself, we use
            xtest = np.array(
                [
                    encode(arch, encoding_type=self.encoding_type, ss_type=self.ss_type)
                    for arch in xtest
                ]
            )
            if self.zc:
                mean, std = -10000000.0, 150000000.0
                xtest = [[*x, (info[i] - mean) / std] for i, x in enumerate(xtest)]
            xtest = np.array(xtest)

        else:
            # when used in aug_lcsvr we feed in ndarray directly
            xtest = xtest

        test_data = self.get_dataset(xtest)
        return np.squeeze(self.model.predict(test_data)) * self.std + self.mean

    def set_random_hyperparams(self):
        if self.hyperparams is None:
            # evaluate the default config first during HPO
            self.hyperparams = self.default_hyperparams.copy()
        return self.hyperparams


class LinearRegression(LinearRegressionBase):
    _model_cls = LR

    @property
    def default_hyperparams(self):
        return {}


class RidgeRegression(LinearRegressionBase):
    _model_cls = Ridge

    @property
    def default_hyperparams(self):
        params = {
            "alpha": 1.0,
        }
        return params

    def set_random_hyperparams(self):
        if self.hyperparams is None:
            params = self.default_hyperparams.copy()
        else:
            params = {
                "alpha": np.random.choice([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]),
            }
        self.hyperparams = params
        return params


class LassoRegression(LinearRegressionBase):
    _model_cls = Lasso

    @property
    def default_hyperparams(self):
        params = {
            "alpha": 1.0,
        }
        return params

    def set_random_hyperparams(self):
        if self.hyperparams is None:
            params = self.default_hyperparams.copy()
        else:
            params = {
                "alpha": np.random.choice([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]),
            }
        self.hyperparams = params
        return params
