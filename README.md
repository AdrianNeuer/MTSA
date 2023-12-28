## Dependency

```bash
numpy==1.24.3
torch==1.13.1
statsmodels==0.14.1
pandas==2.0.3
```

## Command

~~~bash
 # TsfKNN
 python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model TsfKNN --decompose True --decomposition MA 
 python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model TsfKNN --decompose True --decomposition STL
 python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model TsfKNN --decompose True --decomposition X11 
 # DLinear
 python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model DL --decompose True --decomposition MA 
 python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model DL --decompose True --decomposition STL
 python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model DL --decompose True --decomposition X11 
 # Theta
 python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model T
 # ARIMA
 python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model 
 # DecompositionModel
 python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model D --decompose True --decomposition MA
 python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model D --decompose True --decomposition STL 
 # ResidualModel
 python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model R --decompose True --decomposition MA
 # Ensemble
 python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model E --decompose True --decomposition MA 
~~~

## Dataset
- **ETTh1.csv**