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
    epsilon = 1e-10
    mape = 100 * np.mean(np.abs((target - predict) / target + epsilon))
    return mape


def smape(predict, target):
    epsilon = 1e-10
    smape = 100 * np.mean(2 * np.abs(target - predict) /
                          (np.abs(predict) + np.abs(target) + epsilon))
    return smape


def mase(predict, target):
    epsilon = 1e-10
    scale = np.mean(np.abs(target[:,24:] - target[:,:-24]))
    errors = np.mean(np.abs(target- predict))
    mase = errors /(scale + epsilon)
    return mase
