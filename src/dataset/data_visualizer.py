import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def data_visualize(dataset, t):
    """
    Choose t continous time points in data and visualize the chosen points. Note that some datasets have more than one
    channel.
    param:
        dataset: dataset to visualize
        t: the number of timestamps to visualize
    """

    data = dataset.data.squeeze(0)
    data_cols = dataset.data_cols
    timestamp = dataset.data_stamp

    index = np.random.choice(np.arange(data.shape[1] - t))
    visual_data = data[:, index: index + t, :]
    visual_stamp = timestamp[index: index + t]
    for i in range(data.shape[2]):
        channel = data_cols[i]
        plt.plot(visual_stamp, visual_data[:, i], label=channel)
        plt.savefig('imgs/'+channel+'.png')
