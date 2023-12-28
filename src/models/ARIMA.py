import numpy as np
import torch

from src.models.base import MLForecastModel
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.eval_measures import aic
from src.utils.decomposition import moving_average, differential_decomposition, STL_decomposition, X11_decomposition


def get_decomposition(args):
    distance_dict = {
        'MA': moving_average,
        'Diff': differential_decomposition,
        'STL': STL_decomposition,
        'X11': X11_decomposition
    }
    return distance_dict[args.decomposition]


def get_order(sequence):
    best_aic = float('inf')
    best_order_aic = None
    for p in range(3):
        for d in range(2):
            for q in range(3):
                model = ARIMA(sequence, order=(p, d, q)).fit()

                current_aic = model.aic

                if current_aic < best_aic:
                    best_aic = current_aic
                    best_order_aic = (p, d, q)
    return best_order_aic


class Arima(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.decompose = args.decompose
        self.decomposition = get_decomposition(args)

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        samples, _, channels = X.shape
        result = np.zeros((samples, pred_len, channels))
        orders = []
        # result1 = np.zeros((samples, pred_len, channels))
        # result2 = np.zeros((samples, pred_len, channels))
        # order = (2, 1, 1)
        # trend, seasonal = self.decomposition(torch.from_numpy(X.copy()))
        # trend, seasonal = trend.numpy(), seasonal.numpy()

        for i in range(channels):
            order = get_order(X[0, :, i])
            orders.append(order)

        for i in range(samples):
            for j in range(channels):
                model = ARIMA(X[i, :, j], order=orders[j]).fit()
                forecast = model.get_forecast(steps=pred_len)
                result[i, :, j] = np.array(forecast.predicted_mean)
                # model = ARIMA(seasonal[i, :, j], order=order).fit()
                # forecast = model.get_forecast(steps=pred_len)
                # result2[i, :, j] = np.array(forecast.predicted_mean)
        return result
