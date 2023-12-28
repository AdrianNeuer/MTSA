# Homework 3

Homework 3 considers multivariate time series (multi input multi output). The datasets used here are `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Electricity`, `Traffic`, `Weather`, `Exchange`, `ILI`.
## Usage examples

```
python main.py --data_path ./dataset/ETT/ETTh1.csv --dataset ETT --model MeanForecast
```

```
python main.py --data_path ./dataset/ETT/ETTh1.csv --dataset ETT --model TsfKNN --n_neighbors 1 --msas MIMO --distance euclidean
```

## Part 1 Decomposition (20 pts)
path: `src/utils/decomposition.py`

**Objective:** Implement STL and X11 decomposition methods to separate the trend and seasonal components from the original time series data.


## Part 2 Model (20 pts)

path: `src/models/ARIMA.py`
path: `src/models/ThetaMethod.py`

**Objective:** Implement the ARIMA and Theta forecasting models.

## Part 3 ResidualModel (60 pts)

path: `src/models/ResidualModel.py`

**Objective:**
The objective is to create an ensemble forecasting model that integrates various individual 
models you've implemented earlier, such as LR, ETS, DLinear, TSFKNN, ARIMA, and ThetaMethod. 
The aim is to leverage the strengths of each model for improved forecasting accuracy.

**Instructions:**

**1. Decomposition-Based Forecasting:**

Implement time series decomposition to separate trend and seasonal components from the original data.
Apply different forecasting models to predict the trend and seasonal components independently.
Combine these predictions to form a comprehensive forecast.

**2. Residual Network Approach:**

Employ a residual network strategy, inspired by [N-Beats](https://arxiv.org/pdf/1905.10437.pdf), which uses multiple MLPs, and each MLP in the network aims to predict the residuals (errors) of the preceding MLP.
For example, the residual from a TSFKNN prediction could be modeled using DLinear.
The final forecast is the cumulative sum of predictions from all models.

**3. Diverse Prediction Methods:**

Experiment with various forecasting methods, such as recursive, non-recursive, direct, and indirect approaches.

**4. Combining Methods for Enhanced Accuracy:**

You can combine the above methods to get a better result, not necessary to use all of them.


## Part 4 Evaluation

**Instructions:**

**1. Apply your ResidualModels to the datasets specified at the start of this project:**

Tips: You can choose the best model on one dateset and use it to predict the other datasets.

The experimental settings used here are the same as [TimesNet](https://arxiv.org/abs/2210.02186). You can easily compare your model with past SOTA models.
If your model is better than SOTA, you can get 15 pts extra.

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

 

## Submission

**1. Modified Code:**

- Provide the modified code for all components of the task.
- Include a `README.md` file in Markdown format that covers the entire task. This file should contain:
  - how to install any necessary dependencies for the entire task.
  - how to run the code and scripts to reproduce the reported results.
  - datasets used for testing in all parts of the task.

**2. PDF Report:**

- Create a detailed PDF report that encompasses the entire task. The report should include sections for each component of the task.

**3. Submission Format:**

- Submit the entire task, including all code and scripts, along with the `README.md` file and the PDF report, in a compressed archive (.zip).

**4. Submission Deadline:**
  2024-01-15 23:55


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