import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch

from src.models.base import MLForecastModel
from src.utils.distance import euclidean, manhattan, chebyshev
from src.utils.decomposition import moving_average, differential_decomposition, STL_decomposition, X11_decomposition


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
        'Diff': differential_decomposition,
        'STL': STL_decomposition,
        'X11': X11_decomposition
    }
    return distance_dict[args.decomposition]


class TsfKNN(MLForecastModel):
    def __init__(self, args):
        self.k = args.n_neighbors
        if args.distance == 'euclidean':
            self.distance = euclidean
        self.msas = args.msas
        self.decompose = args.decompose
        self.decomposition = get_decomposition(args)
        self.decompose_x = None
        self.individual = args.individual
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        self.X = X[0, :, :]

    def _search(self, x, X_s, seq_len, pred_len):
        if self.msas == 'MIMO':
            if self.individual:
                channels = X_s.shape[2]
                fore_i = []
                for i in range(channels):
                    distances = np.sqrt(np.sum((np.expand_dims(x[:, i],axis=1).T-X_s[:, :seq_len, i])**2, axis=1))
                    indices_of_smallest_k = np.argsort(distances)[:self.k]
                    neighbor_fore = X_s[indices_of_smallest_k, seq_len:, i]
                    x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
                    fore_i.append(np.expand_dims(x_fore,axis=-1))
                fore_i = np.concatenate(fore_i, axis=-1)
                return fore_i
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
        for i in range(X.shape[0]):
            x = X[i, :, :]
            x_fore = self._search(x, X_s, seq_len, pred_len)
            fore.append(x_fore)
        fore = np.concatenate(fore, axis=0)
        return fore