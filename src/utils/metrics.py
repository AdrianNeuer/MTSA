import numpy as np

"""
target : (samples, pred_len)
"""


def mse(predict, target):
    return np.mean((target - predict) ** 2)


# TODO: implement the metrics
def mae(predict, target):
    mae = np.mean(np.abs(target - predict))
    return mae


def mape(predict, target):
    epsilon = 1e-3
    mape = 100 * np.mean(np.abs((target - predict) / (target + epsilon)))
    return mape


def smape(predict, target):
    epsilon = 1e-3
    smape = 100 * np.mean(2 * np.abs(target - predict) /
                          (np.abs(predict) + np.abs(target) + epsilon))
    return smape


def mase(predict, target):
    epsilon = 1e-3
    scale = np.mean(np.abs(target[:, 24:] - target[:, :-24]), axis=1) + epsilon
    errors = np.mean(np.abs(target - predict), axis=1)
    mase = np.mean(errors / scale)
    return mase
