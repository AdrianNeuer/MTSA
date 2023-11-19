# Report
## Part 1
与 HW1 中的实现类似，在transform类中记录计算到的均值和方差，以便后续还原数据。
## Part 2
- Distance:实现了曼哈顿距离和切比雪夫距离，与已经实现的欧几里得距离类似，最后返回包含所有sample之间距离的一维向量。
- Temporal embedding:
  - lag-based: 在timestamps固定为96的基础上，每隔lag步采一次样，得到我们的embedding。
  - fourier：对所有数据做快速傅立叶变换，尝试过截取低频片段作为embedding与原始片段差别不大，故直接将原始信号作为embedding。
  - Autoencoder：采用单个线性层分别作为encoder和decoder，可以自己设定encoding_size，利用fit中传入的数据，sliding_window之后传入Autoencoder中训练，利用训练得到的encoder对预测数据和训练数据进行编码，得到的embedding作为计算距离的向量
  - 实验结果如下：
    | Temporal Embedding            | Distance  | MSE   | MAE  |
    | ----------------------------- | --------- | ----- | ---- |
    | lag-based (lag=1)             | euclidean | 13.51 | 2.21 |
    | lag-based (lag=1)             | manhattan | 13.51 | 2.17 |
    | lag-based (lag=1)             | chebyshev | 14.30 | 2.36 |
    | lag-based (lag=2)             | euclidean | 14.18 | 2.25 |
    | lag-based (lag=2)             | manhattan | 13.91 | 2.20 |
    | lag-based (lag=2)             | chebyshev | 14.72 | 2.39 |
    | lag-based (lag=3)             | euclidean | 14.08 | 2.24 |
    | lag-based (lag=3)             | manhattan | 13.78 | 2.19 |
    | lag-based (lag=3)             | chebyshev | 14.99 | 2.39 |
    | lag-based (lag=8)             | euclidean | 16.64 | 2.37 |
    | lag-based (lag=8)             | manhattan | 15.76 | 2.32 |
    | lag-based (lag=8)             | chebyshev | 16.92 | 2.43 |
    | Fourier                       | euclidean | 42.56 | 3.62 |
    | Fourier                       | manhattan | 41.66 | 3.74 |
    | Fourier                       | chebyshev | 43.17 | 4.19 |
    | Autoencoder(encoding_size=48) | euclidean | 14.02 | 2.24 |
    | Autoencoder(encoding_size=48) | manhattan | 14.42 | 2.28 |
    | Autoencoder(encoding_size=48) | chebyshev | 16.09 | 2.64 |
    | Autoencoder(encoding_size=24) | euclidean | 13.88 | 2.23 |
    | Autoencoder(encoding_size=24) | manhattan | 14.26 | 2.24 |
    | Autoencoder(encoding_size=24) | chebyshev | 15.02 | 2.47 |
    | Autoencoder(encoding_size=32) | euclidean | 13.85 | 2.25 |
    | Autoencoder(encoding_size=32) | manhattan | 14.46 | 2.26 |
    | Autoencoder(encoding_size=32) | chebyshev | 14.98 | 2.54 |

## Part 3
- 参考所给论文实验中进行的DLinear复现，将输入进行分解后分别进入两个线性层，得到趋势和季节的输出后将二者相加得到还原后的输出，利用所给的训练数据多次训练之后对测试数据进行预测。

## Part 4
- Moving_Average分解：滑动平均采用了一个平均池化核，对于当前时间点，采用其前后总共 seasonal_period 个时间点进行平均求解得到趋势项，因此还需要对序列进行补齐。
- Diff分解，采用一阶差分作为趋势项，剩余项作为季节项。
- 实验结果如下：
    | Model   | Decomposition | MSE   | MAE  |
    | ------- | ------------- | ----- | ---- |
    | TsfKNN  | MA            | 15.95 | 2.15 |
    | DLinear | MA            | 6.16  | 1.33 |
    | TsfKNN  | Diff          | 17.08 | 2.21 |
    | DLinear | Diff          | 6.17  | 1.34 |