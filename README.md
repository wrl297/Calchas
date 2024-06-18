# Calchas

## Introduction

This is the datasets and implementation of Calchas as described in "Removing Obstacles before Breaking Through the Memory Wall: A Close Look at HBM Errors in the Field," presented at USENIX ATC'24. Calchas is a hierarchical, comprehensive, and non-intrusive failure prediction framework for HBM. 

## Description of datasets

To encourage researchers to explore the characteristics of HBM failures,we release datasets collect from 19 data centers. The datasets are contained in the Data folder, divided into two parts:

&nbsp;&nbsp;&nbsp;&nbsp;● **processed_data** includes four CSV files with features and labels generated from different hierarchical levels: `data_for_bank-level_prediction.csv`, `data_for_col-level_prediction.csv`, `data_for_row-level_prediction.csv`, and `data_for_server-level_prediction.csv`. For instance, `data_for_bank-level_prediction` is the data used for predictions at the bank level, as shown in the example below:

| Peak Power  | Aver Power  | Temp        | CE_Row | CE_Col | CE_Cell | UER_Row | UER_Col | UER_Cell | UEO_Row | UEO_Col | UEO_Cell | All_Row | All_Col | All_Cell | SID_0 | SID_1 | label |
| ----------- | ----------- | ----------- | ------ | ------ | ------- | ------- | ------- | -------- | ------- | ------- | -------- | ------- | ------- | -------- | ----- | ----- | ----- |
| 1           | 1           | 1           | 1      | 1      | 1       | 0       | 0       | 0        | 0       | 0       | 0        | 1       | 1       | 1        | 1     | 0     | 0     |
| 1.036677418 | 1.035688311 | 0.992300485 | 1      | 1      | 1       | 0       | 0       | 0        | 0       | 0       | 0        | 1       | 1       | 1        | 1     | 0     | 0     |

&nbsp;&nbsp;&nbsp;&nbsp;● **raw_data** contains only one CSV file, which includes specific information about the location, time, and type of errors. Examples of the data format is shown below:

| Datacenter  | Server      | Name | Stack | SID  | PcId | BankGroup | BankArray | Col  | Row    | Time       | EccType |
| ----------- | ----------- | ---- | ----- | ---- | ---- | --------- | --------- | ---- | ------ | ---------- | ------- |
| Datacenter8 | 0.108.38.22 | DSA3 | 0x3   | 0x0  | 0x1  | 0x2       | 0x1       | 0x54 | 0x3e2b | 1650690000 | UER     |
| Datacenter8 | 0.108.38.22 | DSA3 | 0x3   | 0x0  | 0x1  | 0x2       | 0x1       | 0x5c | 0x3fbb | 1650690000 | UER     |
| Datacenter0 | 0.0.0.16    | DSA8 | 0x0   | 0x0  | 0x4  | 0x2       | 0x3       | 0x58 | 0x2a57 | 1652709600 | CE      |

__Note that__ we have anonymized some information to avoid sensitive information being inferred.

## Analyses and Prediction 

The following instructions will guide you on running the prediction code on your local machine.

### Prerequisites 

To run this project, please ensure that your system has Python 3.6 or a newer version installed. Then, execute the following command to install the required libraries:

```
pip3 install -r requirements.txt
```



## Source Code Structure 

Our code is divided into two parts:

- **Analyses**:  Contains nine files that analyze different characteristics of errors.
  - `Avg_temp_distribution.py` 
  - `Ce_storm_machine.py` 
  - `Dataset_analyze.py` 
  - `Error_mode.py` 
  - `Max_temp_distribution.py` 
  - `Power_impact.py` 
  - `Spatial_locality.py` 
  - `Structure_impact.py` 
  - `Time_between_error.py`
- **Prediction**: Contains four files that conduct experiments on the performance under different settings of __Calchas__.
- `Prediction_performance.py`

- `Diff_model.py` 

- `Diff_observation_window.py` 

- `Diff_prediction_window.py` 

**Please note** that the file names represent the type of analysis or prediction. For example, `Prediction_performance.py`represents the preformance of __Calchas__.

## Run

If you want to make predictions, please execute the following command:

```
cd <folder>
python3 <filename>.py
```

For example, if you want to check the performance of __Calchas__ , you can execute the following commands:

```
cd Experiments
python3 Prediction_performance.py
```

Subsequently, you can obtain the following output on the console:

```
=======Test1 for each predictor=======

Results of row-level predictor (Precision, Recall, F1_score)
RF with threshold=0.55: 0.6979166666666666, 0.881578947368421, 0.7790697674418604
Default RF: 0.53125, 0.8947368421052632, 0.6666666666666666

Results of col-level predictor (Precision, Recall, F1_score)
RF with threshold=0.6: 0.7267080745341615, 0.8666666666666667, 0.7905405405405406
Default RF: 0.7166666666666667, 0.9555555555555556, 0.8190476190476191

Results of bank-level predictor (Precision, Recall, F1_score)
RF with threshold=0.55: 0.6681034482758621, 0.7380952380952381, 0.7013574660633485
Default RF: 0.6681034482758621, 0.7380952380952381, 0.7013574660633485

Results of server-level predictor (Precision, Recall, F1_score)
RF with threshold=0.6: 0.3325581395348837, 0.5674603174603174, 0.4193548387096774
Default RF: 0.2826510721247563, 0.5753968253968254, 0.3790849673202614
```

Since the prediction model uses machine learning models, the results of  prediction may vary. 

## Citation

Please cite our paper if you use this dataset.

```
@inproceedings {wu2024, 
title = {Removing Obstacles before Breaking Through the Memory Wall: A Close Look at HBM Errors in the Field}, 
author = {Wu, Ronglong and Zhou, Shuyue and Lu, Jiahao and Shen, Zhirong and Xu, Zikang and Shu, Jiwu and Yang, Kunlin and Lin, Feilong and Zhang, Yiming} 
booktitle = {2024 USENIX Annual Technical Conference (USENIX ATC 24)}, 
year = {2024} 
}
```

