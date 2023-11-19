# Dependencies
- python==3.9
- numpy==1.24.3
- pytorch==1.13.1

# Dataset
实验采用数据集为 ETT-small/ETTh1.csv
# How to run the code
- 运行 lag-based embedding 的命令如下：
  ```bash
  python --data_path ./dataset/ETT-small/ETTh1.csv --model TsfKNN --distance d --lag x
  ```
  d 可以为三种不同的距离，x 表示采用lag的大小。
- 运行傅立叶变换产生的 embedding 命令如下：
  ```bash
  python --data_path ./dataset/ETT-small/ETTh1.csv --model TsfKNN --distance d --fourier True
  ```
  d 可以为三种不同的距离。
- 运行 Autoencoder 作为 embedding 的命令如下：
  ```bash
  python --data_path ./dataset/ETT-small/ETTh1.csv --model TsfKNN --distance d --auto True --encoding_size x
  ```
  d 为三种不同的距离，x 表示 embedding size。
- 将分解方法与两种模型结合实验的命令如下：
  ```bash
  python --data_path ./dataset/ETT-small/ETTh1.csv --model m --transform ST --decompose True --decomposition de
  ```
  m 表示 TsfKNN, DL 两种模型，de 表示两种分解方法(MA, Diff)。
