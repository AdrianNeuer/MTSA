import numpy as np


def mse(predict, target):
    return np.mean((target - predict) ** 2)


# TODO: implement the metrics
def mae(predict, target):
    mae = np.mean(np.abs(target - predict))
    return mae


def mape(predict, target):
    mape = 100 * np.mean(np.abs((target - predict) / target))
    return mape


def smape(predict, target):
    smape = 100 * np.mean(2 * np.abs(target - predict) /
                          (np.abs(predict) + np.abs(target)))
    return smape


def mase(predict, target):
    raise NotImplementedError
