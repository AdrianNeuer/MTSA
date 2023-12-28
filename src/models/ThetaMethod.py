import numpy as np
import pandas as pd
import torch


from src.models.base import MLForecastModel
from statsmodels.tsa.forecasting.theta import ThetaModel
from src.utils.decomposition import moving_average, differential_decomposition, STL_decomposition, X11_decomposition


def get_decomposition(args):
    distance_dict = {
        'MA': moving_average,
        'Diff': differential_decomposition,
        'STL': STL_decomposition,
        'X11': X11_decomposition
    }
    return distance_dict[args.decomposition]


class ThetaMethod(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.decompose = args.decompose
        self.decomposition = get_decomposition(args)

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        index = pd.date_range(start='2023-01-01', periods=X.shape[1])
        samples, _, channels = X.shape
        result = np.zeros((samples, pred_len, channels))
        # result1 = np.zeros((samples, pred_len, channels))
        # result2 = np.zeros((samples, pred_len, channels))
        # trend, seasonal = self.decomposition(torch.from_numpy(X.copy()))
        # trend, seasonal = trend.numpy(), seasonal.numpy()
        for i in range(samples):
            for j in range(channels):
                model = ThetaModel(
                    pd.Series(X[i, :, j], index=index)).fit()
                forecast = model.forecast(steps=pred_len)
                result[i, :, j] = forecast.values
                # model = ThetaModel(
                #     pd.Series(seasonal[i, :, j], index=index)).fit()
                # forecast = model.forecast(steps=pred_len)
                # result2[i, :, j] = forecast.values
        return result
