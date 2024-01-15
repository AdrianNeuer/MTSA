import numpy as np
import torch.nn as nn
from torch import optim
import torch
from numpy.lib.stride_tricks import sliding_window_view
import torch
from src.models.DLinear import Model
from src.models.base import MLForecastModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def spirit(X):
    samples = X.shape[0]
    T = X.shape[1]
    channels = X.shape[2]
    k = 1
    E = 0
    E1 = [0 for i in range(channels)]
    fE = 0.95
    FE = 0.98
    lamda = 0.96
    W = np.identity(channels, dtype=float)
    d = np.ones(channels) * 0.1

    for i in range(T):
        x = X[:, i, :]
        # print(i)
        for j in range(k):
            yj = (np.expand_dims(W[:, j], axis=1).T @ x.T)  # (1, samples)
            d[j] = lamda * d[j] + np.sum(yj**2)
            ej = x - yj.T @ np.expand_dims(W[:, j].T, axis=0) # (samples, channels)
            W[:, j] = W[:, j] + 1/d[j] * np.mean(yj.T * ej, axis=0) # (channels, )
            x = x - yj.T * np.expand_dims(W[:, j], axis=0) # (samples, channels)
            
            E1[j] = (i*E1[j] + np.sum(yj**2)) / (i+1)
        E = (i*E + np.sum(np.sum(x**2, axis=1), axis=0))/(i+1)

        Ek = np.cumsum(E1)[k-1]
        print(Ek/E)
        if Ek < fE * E:
            k+=1
        if Ek > FE * E:
            k-=1
    return W, k



class SPIRIT(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.channels = args.channels
        self.model = Model(args).to(device)
        self.optim = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.epoch = 5
        self.batch_size = 64
        self.W = None
        self.k = 0

    def _fit(self, X: np.ndarray) -> None:
        self.X = X[0, :, :]
        channels = X.shape[2]
        X_s = sliding_window_view(
            self.X, (self.seq_len + self.pred_len, channels)).reshape(-1, self.seq_len + self.pred_len, channels)

        self.W, self.k = spirit(X_s)

        new_X = (self.W[:, :self.k].T @ X_s.transpose(0,2,1)).transpose(0,2,1)
        print(new_X.shape)

        train_X = torch.from_numpy(
            new_X[:, :self.seq_len, :]).to(device)
        train_Y = torch.from_numpy(
            new_X[:, self.seq_len:, :]).to(device)

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
