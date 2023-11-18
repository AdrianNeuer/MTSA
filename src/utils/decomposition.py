import numpy as np
import torch
import torch.nn as nn


def moving_average(x, seasonal_period=25):
    """
    Moving Average Algorithm
    Args:
        x (numpy.ndarray): Input time series data
        seasonal_period (int): Seasonal period
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
    """
    # kernel_size = 25
    new_x = x.clone()
    avg = nn.AvgPool1d(kernel_size=seasonal_period, stride=1, padding=0)
    front = x[:, 0:1, :].repeat(1, (seasonal_period - 1) // 2, 1)
    end = x[:, -1:, :].repeat(1, (seasonal_period - 1) // 2, 1)
    x = torch.cat([front, x, end], dim=1)
    x = avg(x.permute(0, 2, 1))
    x = x.permute(0, 2, 1)

    return new_x - x, x


def differential_decomposition(x):
    """
    Differential Decomposition Algorithm
    Args:
        x (numpy.ndarray): Input time series data
    Returns:
        trend (numpy.ndarray): Trend component
        seasonal (numpy.ndarray): Seasonal component
    """

    trend = torch.diff(x, axis=1)
    trend = nn.functional.pad(trend, (0, 0, 1, 0))

    return x - trend, trend
