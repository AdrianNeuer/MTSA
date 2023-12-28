# Report
## 502023370016 黄彦骁

### Decomposition
- 对于STL分解，针对每条sample每个channel独立进行分解，由于自己实现的分解方法太慢且参数无法确定，调用statsmodel接口进行分解
- 对于X11分解，同样针对每个独立序列进行分解，调用statsmodel相关接口进行实现

### ARIMA
- 由于定阶时间过长，因此针对每一个channel，取第一个sample定阶作为整个channel的阶数，同时由于有时阶数p+q过大，会导致内置的LU分解报错，因此定阶采取的p,q都不大。
  在fit过程中不采取任何操作，直接用测试的X进行预测。

### ThetaModel
- 对于每一个sample，将index设为从2023.01.01开始的时间戳，与数据一起作为Series给模型进行预测，与ARIMA类似，fit过程不操作，直接用测试的X进行预测。

### Decomposition-based model
- 由于模型的特性，采用ARIMA预测季节项和ThetaModel预测趋势项的组合，能获得比二者更好的结果。

### ResidualModel
- 采用DLinear模型预测TsfKNN的残差，将训练数据切片，针对每一个切片后得到的sample，在数据找与其距离第二近的sample，将这个sample作为模型预测的backcast，与原始sample相减得到DLinear的残差训练数据，将DLinear预测的残差与TsfKNN的结果相加，最后能够减少TsfKNN的原始误差。

### EnsembleModel
- 将上述四种模型，按照预测精确度对输出预测值进行加权求和，作为Ensemble的预测。

### Performance

| Dataset | pred_len | Models                       | Decomposition | MSE   | MAE   |
| ------- | -------- | ---------------------------- | ------------- | ----- | ----- |
| ETTh1   | 96       | TsfKNN                       | MA            | 1.337 | 0.871 |
| ETTh1   | 96       | TsfKNN                       | STL           | 1.518 | 0.965 |
| ETTh1   | 96       | TsfKNN                       | X11           | 1.552 | 0.978 |
| ETTh1   | 96       | DLinear                      | MA            | 0.559 | 0.534 |
| ETTh1   | 96       | DLinear                      | STL           | 0.560 | 0.536 |
| ETTh1   | 96       | DLinear                      | X11           | 0.581 | 0.551 |
| ETTh1   | 96       | Theta                        | None          | 1.544 | 0.909 |
| ETTh1   | 96       | ARIMA                        | None          | 0.815 | 0.683 |
| ETTh1   | 96       | Decomposition(ARIMA + Theta) | MA            | 0.719 | 0.543 |
| ETTh1   | 96       | Decomposition(ARIMA + Theta) | STL           | 0.732 | 0.568 |
| ETTh1   | 96       | Residual                     | MA            | 1.227 | 0.862 |
| ETTh1   | 96       | Ensemble                     | MA            | 0.568 | 0.555 |

