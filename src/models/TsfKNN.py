import numpy as np
import torch
import torch.nn as nn
from torch import optim
from numpy.lib.stride_tricks import sliding_window_view

from src.models.base import MLForecastModel
from src.utils.distance import euclidean, manhattan, chebyshev
from src.utils.decomposition import moving_average, differential_decomposition


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_distance(args):
    distance_dict = {
        'euclidean': euclidean,
        'manhattan': manhattan,
        'chebyshev': chebyshev,
    }
    return distance_dict[args.distance]


def get_decomposition(args):
    distance_dict = {
        'MA': moving_average,
        'Diff': differential_decomposition
    }
    return distance_dict[args.decomposition]


def lagbased(data, tau):
    return data[:, ::tau, :]


def fourier(data):
    return np.abs(np.fft.fft(data, axis=1))


class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, encoding_size)
        self.decoder = nn.Linear(encoding_size, input_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        return x


class TsfKNN(MLForecastModel):
    def __init__(self, args):
        self.k = args.n_neighbors
        self.distance = get_distance(args)
        self.msas = args.msas
        self.decompose = args.decompose
        self.decomposition = get_decomposition(args)
        self.lag = args.lag
        self.fourier = args.fourier
        self.auto = args.auto
        self.encoding_size = args.encoding_size
        self.autoencoder = Autoencoder(
            args.seq_len, encoding_size=self.encoding_size).to(device)
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        self.X = X[0, :, :]

    def _search(self, x, X_s, seq_len, pred_len):
        if self.msas == 'MIMO':
            if self.decompose:
                de_x = np.expand_dims(x, axis=0)
                seasonal_x, trend_x = self.decomposition(
                    torch.from_numpy(de_x))
                new_x = np.concatenate(
                    (trend_x.numpy(), seasonal_x.numpy()), axis=1)
                de_X = X_s[:, :seq_len, :].copy()
                seasonal_X, trend_X = self.decomposition(
                    torch.from_numpy(de_X))
                new_X = np.concatenate(
                    (trend_X.numpy(), seasonal_X.numpy()), axis=1)
                distances = self.distance(new_x, new_X)
            else:
                distances = self.distance(x, X_s[:, :seq_len, :])
            indices_of_smallest_k = np.argsort(distances)[:self.k]
            neighbor_fore = X_s[indices_of_smallest_k, seq_len:, :]
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
            return x_fore

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        fore = []
        bs, seq_len, channels = X.shape
        X_s = sliding_window_view(
            self.X, (seq_len + pred_len, channels)).reshape(-1, seq_len + pred_len, channels)
        if self.lag != 0:
            X = lagbased(X, self.lag)
            X_s = np.concatenate(
                (lagbased(X_s[:, :seq_len, :], self.lag), X_s[:, seq_len:, :]), axis=1)
            seq_len = seq_len // self.lag
        if self.fourier:
            X = fourier(X)
            X_s = np.concatenate(
                (fourier(X_s[:, :seq_len, :]), X_s[:, seq_len:, :]), axis=1)
        if self.auto:
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.01)
            batch_size = 64
            epochs = 10
            data = torch.from_numpy(X_s[:, :seq_len, :].copy()).to(device)
            for epoch in range(epochs):
                total_loss = 0
                for i in range(0, data.shape[0], batch_size):
                    train_x = data[i: i + batch_size].to(torch.float32)
                    pred = self.autoencoder(train_x)

                    loss = criterion(train_x, pred)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print("Epoch : {}, Loss : {}".format(epoch+1, total_loss))
            X = self.autoencoder.encoder(torch.from_numpy(X).to(device).to(
                torch.float32).permute(0, 2, 1)).cpu().permute(0, 2, 1).detach().numpy()
            X_s_pre = self.autoencoder.encoder(torch.from_numpy(X_s[:, :seq_len, :].copy()).to(
                device).to(torch.float32).permute(0, 2, 1)).permute(0, 2, 1).cpu().detach().numpy()
            X_s = np.concatenate((X_s_pre, X_s[:, seq_len:, :]), axis=1)
            seq_len = self.encoding_size

        for i in range(X.shape[0]):
            x = X[i, :, :]
            x_fore = self._search(x, X_s, seq_len, pred_len)
            fore.append(x_fore)
        fore = np.concatenate(fore, axis=0)
        return fore
