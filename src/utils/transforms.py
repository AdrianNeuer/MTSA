import numpy as np


class Transform:
    """
    Preprocess time series
    """

    def transform(self, data):
        """
        :param data: raw timeseries
        :return: transformed timeseries
        """
        pass

    def inverse_transform(self, data):
        """
        :param data: raw timeseries
        :return: inverse_transformed timeseries
        """
        pass


class IdentityTransform(Transform):
    def __init__(self, args):
        pass

    def transform(self, data):
        pass

    def inverse_transform(self, data):
        pass
# TODO: add other transforms


class NormalizationTransform(Transform):
    def __init__(self, args):
        self.Max = None
        self.Min = None

    def transform(self, data):
        """
        data --> train_data : (n_samples, timestamp, channel)
        """
        self.Max = np.max(data, axis=1)
        self.Min = np.min(data, axis=1)
        normalize_data = np.where(
            self.Max != self.Min, (data - self.Min)/(self.Max - self.Min), 0.0)
        return normalize_data

    def inverse_transform(self, data):
        return data * (self.Max - self.Min) + self.Min


class StandardizationTransform(Transform):
    def __init__(self, args):
        self.mu = None
        self.sigma = None

    def transform(self, data):
        self.mu = np.mean(data, axis=1)
        self.sigma = np.std(data, axis=1)
        normalize_data = np.where(
            self.sigma != 0, (data-self.mu)/self.sigma, 0.0)
        return normalize_data

    def inverse_transform(self, data):
        return data * self.sigma + self.mu


class MeanNormalizationTransform(Transform):
    def __init__(self, args):
        self.Max = None
        self.Min = None
        self.mu = None

    def transform(self, data):
        self.Max = np.max(data, axis=1)
        self.Min = np.min(data, axis=1)
        self.mu = np.mean(data, axis=1)
        normalize_data = np.where(
            self.Max != self.Min, (data - self.mu)/(self.Max - self.Min), 0.0)
        return normalize_data

    def inverse_transform(self, data):
        return data * (self.Max - self.Min) + self.mu


class BoxCosTransform(Transform):
    def __init__(self, args):
        self.lamda1 = args.box_lambda
        self.lamda2 = None

    def transform(self, data):
        if np.any(data <= 0):
            self.lamda2 = abs(np.min(data, axis=1)) + 1
            data += self.lamda2
        if self.lamda1 == 0:
            normalize_data = np.log(data)
        else:
            normalize_data = np.power(data, self.lamda1) / self.lamda1

        return normalize_data

    def inverse_transform(self, data):
        if self.lamda1 == 0:
            inverse_data = np.exp(data)
        else:
            inverse_data = np.power(data * self.lamda1, 1/self.lamda1)
        if self.lamda2 is not None:
            inverse_data -= self.lamda2
        return inverse_data
