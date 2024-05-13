# Calchas

## Introduction

This is the implementation of Calchas described in our paper "Removing Obstacles before Breaking Through the Memory Wall: A Close Look at HBM Errors in the Field" appeared in USENIX ATC'24. Calchas is a hierarchical, comprehensive, and non-intrusive failure prediction framework for HBM. Please contact xxxxxx  if you have any questions.



## Getting Started 

The following instructions will guide you on how to run the codes on your local machine.

### Prerequisites 

To run this project, ensure that your system is equipped with Python 3.6 or higher version, along with the specified versions of the following libraries: scikit-learn (version 0.24.2), matplotlib (version 3.3.4), numpy (version 1.19.5), and pandas (version 0.25.3). This setup has been tested and confirmed to work on both Linux and Windows.

## Run

In our platform,  use the following command to access our source code files:

```
cd ATC_Artifact/
```



## Structure 

Our code is divided into three parts, each corresponding to different sections of the article. 

- **Analyses**(Section 3)
  - `Fig2_spatial_locality.py` - Support Finding 1 and Finding 2.
  - `Fig3_structure_impact.py` -Support Finding 3 and Finding 4.
  - `Fig6_time_between_error.py` - Support Finding 6.
  - `Fig7_power_impact.py` - Support Finding 8.
  - `Fig8a_avg_temp_distribution.py` - Support Finding 9.
  - `Fig8b_max_temp_distribution.py` - Support Finding 9.
  - `Tab1_dataset_analyze.py` - Support dataset overview in Table 1.
  - `Tab2_error_mode.py` - Support Finding 5.
  - `Tab3_ce_storm_machine.py` - Support Finding 7.

- **Unsuccessful Attempts**(Section 4)
  - `Fig9a_CE_rate_predict.py` -Support Attempt 1(unsuccessful).
  - `Fig9b_GBDT_predict.py` - Support Attempt 2(unsuccessful).
  - `Fig9b_RF_predict.py` - Support Attempt 2(unsuccessful).
- **Experiments**(Section 5)
  - `Fig12_prediction_performance.py` - Support Exp#1.
  - `Fig13_diff_model.py` - Support Exp#2.
  - `Fig14_diff_observation_window.py` - Support Exp#3.
  - `Fig15_diff_prediction_window.py` - Support Exp#4.

**Note that** the file name prefixes 'Fig' and 'Tab' correspond to figures and tables in the paper, respectively. For example, the `Tab1_dataset_analyze.py`corresponds to the data in Table 1 of the paper, and the `Fig2_spatial_locality.py` corresponds to the results shown in Figure 2 of the paper.

**Analyses**

If you want to view our findings, you need to run the following command:

```
cd Analyses
python3 <filename>.py
```

For example, if you want to run the `Fig2_spatial_locality.py`  to analyze spatial locality of errors, you can do so by executing the following command in your terminal:

```
cd Analyses
python3 Fig2_spatial_locality.py
```

**Note that** all results will be displayed in the terminal.

**Attempts**

If you want to view our attempts, you need to run the following command:

```
cd Attempts
python3 <filename>.py
```

For example, if you want to run the ``Fig9a_CE_rate_predict.py``, you can do so by executing the following command in your terminal:

```
cd Attempts
python3 Fig9a_CE_rate_predict.py
```

All results will be displayed in the terminal.

**Experiments**



 **Note that** the machine learning model may not be entirely robust, leading to fluctuations in prediction results. To mitigate this, we performed each experiment at least five times. Detailed results used in our paper can be found in the **Data** folder.

For example, you can use the following command to view the Exp#1 results used in our paper:

First，enter the **Data** folder:

```
cd /root/ATC_Artifact/Data/
```

Second, enter the **Fig12_prediction** folder:

```
cd Fig12_prediction
cd  result_used_in_paper
```

The tree structure should be like:
    ├── Fig12_input_data.csv
    ├── RF_bank-level_predictor.csv
    ├── RF_col-level_predictor.csv
    ├── RF_row-level_predictor.csv
    └── RF_server-level_predictor.csv
