## Dependency
~~~bash
numpy==1.24.3
scikit-learn==1.3.1
torch==1.13.1
statsmodels==0.14.1
~~~
## Command
~~~bash
--individual # wether independent
--model # T for TsfKNN, D for DLinear, S for SPIRIT.

# ETTh1
python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model D --individual True
python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model D 
python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model T --individual True
python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model T 
python main.py --data_path ./dataset/ETT-small/ETTh1.csv --dataset ETT --model S

# ETTm1
python main.py --data_path ./dataset/ETT-small/ETTm1.csv --dataset ETT --model D --individual True
python main.py --data_path ./dataset/ETT-small/ETTm1.csv --dataset ETT --model D 
python main.py --data_path ./dataset/ETT-small/ETTm1.csv --dataset ETT --model T --individual True
python main.py --data_path ./dataset/ETT-small/ETTm1.csv --dataset ETT --model T 
python main.py --data_path ./dataset/ETT-small/ETTm1.csv --dataset ETT --model S

#illness
python main.py --data_path ./dataset/illness/national_illness.csv --dataset Custom --model D --individual True
python main.py --data_path ./dataset/illness/national_illness.csv --dataset Custom --model D 
python main.py --data_path ./dataset/illness/national_illness.csv --dataset Custom --model T --individual True
python main.py --data_path ./dataset/illness/national_illness.csv --dataset Custom --model T 
python main.py --data_path ./dataset/illness/national_illness.csv --dataset Custom --model S
~~~

## Dataset
ETTh1.csv, ETTm1.csv, national_illness.csv