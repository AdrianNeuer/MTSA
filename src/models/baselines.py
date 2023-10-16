import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from src.models.base import MLForecastModel


class ZeroForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        return np.zeros((X.shape[0], pred_len))


class MeanForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        mean = np.mean(X, axis=-1).reshape(X.shape[0], 1)
        return np.repeat(mean, pred_len, axis=1)

# TODO: add other models based on MLForecastModel


class LinearRegression(MLForecastModel):
    def __init__(self, args):
        super().__init__()

    def _fit(self, X):
        self.X = X[:, :, -1]

    def _forecast(self, X, pred_len):
        seq_len = X.shape[1]
        subseries = np.concatenate(
            ([sliding_window_view(v, seq_len + pred_len) for v in self.X]))
        train_x = subseries[:, :seq_len]
        train_y = subseries[:, seq_len:]
        X_b = np.c_[np.ones((train_x.shape[0], 1)), train_x]
        self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(train_y)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        pred = X_b.dot(self.weights)
        return pred


class ExpotenialSmoothing(MLForecastModel):
    def __init__(self, args):
        super().__init__()
        self.alpha = args.ES_alpha
        self.beta = args.ES_beta

    def _fit(self, X):
        pass

    def _forecast(self, X, pred_len):
        a = np.zeros_like(X)
        b = np.zeros_like(X)

        a[:, 0] = X[:, 0]
        b[:, 0] = X[:, 1] - X[:, 0]
        for i in range(1, X.shape[1]):
            a[:, i] = self.alpha * X[:, i] + \
                (1-self.alpha) * (a[:, i-1]+b[:, i-1])
            b[:, i] = self.beta * (a[:, i] - a[:, i-1]) + \
                (1 - self.beta) * b[:, i-1]
        pred = np.zeros((X.shape[0], pred_len))
        for i in range(pred_len):
            pred[:, i] = a[:, -1] + (i+1)*b[:, -1]

        return pred
