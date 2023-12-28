import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from src.models.base import MLForecastModel
from src.models.ARIMA import Arima
from src.models.ThetaMethod import ThetaMethod
from src.models.DLinear import DLinear
from src.models.TsfKNN import TsfKNN
from numpy.lib.stride_tricks import sliding_window_view
from src.utils.distance import euclidean, manhattan, chebyshev
from src.utils.decomposition import moving_average, differential_decomposition, STL_decomposition, X11_decomposition


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_decomposition(args):
    distance_dict = {
        'MA': moving_average,
        'Diff': differential_decomposition,
        'STL': STL_decomposition,
        'X11': X11_decomposition
    }
    return distance_dict[args.decomposition]


def get_distance(args):
    distance_dict = {
        'euclidean': euclidean,
        'manhattan': manhattan,
        'chebyshev': chebyshev,
    }
    return distance_dict[args.distance]


class ResidualModel(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.k = args.n_neighbors
        self.decompose = args.decompose
        self.decomposition = get_decomposition(args)
        self.decompose_x = None
        self.distance = get_distance(args)
        self.model = ResModel(args).to(device)
        self.optim = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.epoch = 5
        self.batch_size = 64
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len

    def _fit(self, X: np.ndarray) -> None:
        self.X = X[0, :, :]
        input = []
        output = []
        seq_len = self.args.seq_len
        pred_len = self.args.pred_len
        channels = X.shape[-1]

        self.X_s = sliding_window_view(
            self.X, (seq_len + pred_len, channels)).reshape(-1, seq_len + pred_len, channels)
        if self.decompose:
            trend, seasonal = self.decomposition(
                torch.from_numpy(self.X_s[:, :seq_len, :].copy()))
            self.decompose_x = np.concatenate(
                (trend.numpy(), seasonal.numpy()), axis=1)
        for i in range(self.X_s.shape[0]):
            # print(i)
            x = self.decompose_x[i, :, :]
            distances = self.distance(x, self.decompose_x)
            indices_of_smallest_k = np.argsort(distances)[2]
            y_fore = np.expand_dims(self.X_s[i, :seq_len, :] -
                                    self.X_s[indices_of_smallest_k, :seq_len, :], axis=0)
            input.append(y_fore)
            x_fore = np.expand_dims(
                self.X_s[i, seq_len:, :] - self.X_s[indices_of_smallest_k, seq_len:, :], axis=0)
            output.append(x_fore)
        input = np.concatenate(input, axis=0)
        output = np.concatenate(output, axis=0)
        print(input.shape, output.shape)

        train_X = torch.from_numpy(input).to(device)
        train_Y = torch.from_numpy(output).to(device)

        for i in range(self.epoch):
            total_loss = 0
            for j in range(0, train_X.shape[0], self.batch_size):
                data = train_X[j: j + self.batch_size].to(torch.float32)
                label = train_Y[j: j + self.batch_size].to(torch.float32)

                pred = self.model(data)
                loss = self.loss_fn(label, pred)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item()
            print("Epoch : {}, Loss : {}".format(i+1, total_loss))

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        fore = []
        bs, seq_len, channels = X.shape
        for i in range(X.shape[0]):
            x = X[i, :, :]
            de_x = np.expand_dims(x, axis=0)
            seasonal_x, trend_x = self.decomposition(
                torch.from_numpy(de_x))
            new_x = np.concatenate(
                (trend_x.numpy(), seasonal_x.numpy()), axis=1)
            distances = self.distance(new_x, self.decompose_x)
            indices_of_smallest_k = np.argsort(distances)[:self.k]
            neighbor_fore = self.X_s[indices_of_smallest_k, seq_len:, :]
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
            fore.append(x_fore)
        fore = np.concatenate(fore, axis=0)
        test_X = torch.from_numpy(X-fore).to(device)
        pred_y = self.model(test_X.to(torch.float32))
        return pred_y.cpu().detach().numpy() + fore


class ResModel(nn.Module):
    def __init__(self, configs):
        super(ResModel, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        output = self.Linear(x)
        output = output.permute(0, 2, 1)
        return output


class DecompositionModel(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.decomposition = get_decomposition(args)
        self.arima = Arima(args)
        self.theta = ThetaMethod(args)

    def _fit(self, X: np.ndarray) -> None:
        self.X = X[0, :, :]
        self.arima.fit(self.X)
        self.theta.fit(self.X)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        trend, seasonal = self.decomposition(torch.from_numpy(X.copy()))
        trend, seasonal = trend.numpy(), seasonal.numpy()
        fore_t = self.arima.forecast(seasonal, pred_len)
        fore_s = self.theta.forecast(trend, pred_len)
        return fore_t + fore_s


class Ensemble(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.decomposition = get_decomposition(args)
        self.arima = Arima(args)
        self.theta = ThetaMethod(args)
        self.dlinear = DLinear(args)
        self.tsfknn = TsfKNN(args)

    def _fit(self, X: np.ndarray) -> None:
        self.dlinear.fit(X)
        self.theta.fit(X)
        self.tsfknn.fit(X)
        self.arima.fit(X)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        pred1 = self.dlinear.forecast(X, pred_len)
        pred2 = self.theta.forecast(X, pred_len)
        pred3 = self.tsfknn.forecast(X, pred_len)
        pred4 = self.arima.forecast(X, pred_len)

        return 0.6*pred1 + 0.1*pred2+0.1*pred3 + 0.2*pred4
