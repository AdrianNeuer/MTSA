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
        raise NotImplementedError

    def inverse_transform(self, data):
        """
        :param data: raw timeseries
        :return: inverse_transformed timeseries
        """
        raise NotImplementedError


class IdentityTransform(Transform):
    def __init__(self, args):
        pass

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

# TODO: add other transforms

class StandardizationTransform(Transform):
    def __init__(self, args):
        self.mu = None
        self.sigma = None

    def transform(self, data):
        self.mu = np.expand_dims(np.mean(data, axis=1), 1)
        self.sigma = np.expand_dims(np.std(data, axis=1), 1)
        normalize_data = np.where(
            self.sigma != 0, (data-self.mu)/self.sigma, 0.0)
        return normalize_data

    def inverse_transform(self, data):
        return data * self.sigma + self.mu