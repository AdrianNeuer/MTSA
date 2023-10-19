## Necessary dependencies

- python==3.9
- numpy==1.24.3
- pandas==2.0.3
- matplotlib==3.8.0

## How to run the code

- 对于part1的数据集分析部分，在主文件夹下运行：

```bash
python main.py --model Mean --datavisualize True
```
​	绘制的数据特征图片会保存在imgs/下

- 对于part4的模型测试阶段，在主文件夹下分别依次运行如下命令：

~~~bash
python main.py --model LR 
python main.py --model LR --transform Normal
python main.py --model LR --transform Standard
python main.py --model LR --transform MeanNormal
python main.py --model LR --transform Box
python main.py --model ES
python main.py --model ES --transform Normal
python main.py --model ES --transform Standard
python main.py --model ES --transform MeanNormal
python main.py --model ES --transform Box
~~~

​	即可得到所有的error结果

## TsfKNN Distance

- 对TsfKNN在两个数据集illness/natinal_illness.csv和ETT-small/ETTh1.csv上采用不同distance得到error，需要在主文件夹下运行如下命令：

```bash
python main.py --model TsfKNN
python main.py --model TsfKNN --distance manhattan
python main.py --model TsfKNN --distance chebyshev
python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model TsfKNN
python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model TsfKNN --distance mamhattan
python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model TsfKNN --distance chebyshev
```

## Dataset

LR和ES实验所用到的数据集为illness/national_illness.csv

TsfKNN实验用到两个数据集illness/national_illness.csv和ETT-small/ETTh1.csv