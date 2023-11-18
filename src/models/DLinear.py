import torch.nn as nn
from torch import optim
import torch
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.models.base import MLForecastModel
from src.utils.decomposition import moving_average, differential_decomposition

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class DLinear(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = Model(args).to(device)
        self.optim = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.epoch = 5
        self.batch_size = 64
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len

    def _fit(self, X: np.ndarray) -> None:
        subseries = np.concatenate(
            ([sliding_window_view(v, (self.seq_len + self.pred_len, v.shape[-1])) for v in X]))
        train_X = torch.from_numpy(
            subseries[:, 0, :self.seq_len, :]).to(device)
        train_Y = torch.from_numpy(
            subseries[:, 0, self.seq_len:, :]).to(device)

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
        test_X = torch.from_numpy(X).to(device)
        pred_y = self.model(test_X.to(torch.float32))
        return pred_y.cpu().detach().numpy()


def get_decomposition(args):
    distance_dict = {
        'MA': moving_average,
        'Diff': differential_decomposition
    }
    return distance_dict[args.decomposition]


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = individual

        # TODO: implement the following layers
        self.decomposition = get_decomposition(configs)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        seansonal, trend = self.decomposition(x)
        seansonal, trend = seansonal.permute(0, 2, 1), trend.permute(0, 2, 1)
        trend_output = self.Linear_Trend(trend)
        seansonal_output = self.Linear_Trend(seansonal)
        output = trend_output + seansonal_output
        output = output.permute(0, 2, 1)
        return output
