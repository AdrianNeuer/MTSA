import numpy as np
import matplotlib.pyplot as plt


def data_visualize(dataset, t):
    """
    Choose t continous time points in data and visualize the chosen points. Note that some datasets have more than one
    channel.
    param:
        dataset: dataset to visualize
        t: the number of timestamps to visualize
    """

    data = np.squeeze(dataset.data, axis=0)
    data_cols = dataset.data_cols
    timestamp = dataset.data_stamp

    index = np.random.choice(np.arange(data.shape[0] - t))
    visual_data = data[index: index + t, :]
    visual_stamp = timestamp[index: index + t]
    for i in range(data.shape[1]):
        channel = data_cols[i]
        plt.plot(visual_stamp, visual_data[:, i], label=channel)
        plt.savefig('imgs/'+channel+'.png')
