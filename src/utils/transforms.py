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
        return (data - self.Min)/(self.Max - self.Min)

    def inverse_transform(self, data):
        return data * (self.Max - self.Min) + self.Min


class StandardizationTransform(Transform):
    def __init__(self, args):
        self.mu = None
        self.sigma = None

    def transform(self, data):
        self.mu = np.mean(data, axis=1)
        self.sigma = np.std(data, axis=1)
        return (data-self.mu)/self.sigma

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
        return (data - self.mu)/(self.Max - self.Min)

    def inverse_transform(self, data):
        return data * (self.Max - self.Min) + self.mu


class BoxCosTransform(Transform):
    def __init__(self, args):
        self.lamda = args.box_lambda

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data