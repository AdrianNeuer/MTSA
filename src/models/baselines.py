import numpy as np

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
    def __init__(self):
        super().__init__()

    def _fit(self, X):
        """
        X : (n_sample, seq_len + pred_len)
        last channel is target
        """
        fit_x = X[:,:,:-1]
        fit_y = X[:,:,-1]

    def _forecast(self, X, pred_len):
        pass