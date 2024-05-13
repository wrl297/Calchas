# Calchas

## Introduction

This is the implementation of Calchas as described in "Removing Obstacles before Breaking Through the Memory Wall: A Close Look at HBM Errors in the Field," presented at USENIX ATC'24. Calchas is a hierarchical, comprehensive, and non-intrusive failure prediction framework for HBM. For any inquiries, please contact rlwoo@stu.xmu.edu.cn.



## Getting Started 

The following instructions will guide you on running the code on your local machine.

### Prerequisites 

To run this project, please ensure that your system has Python 3.6 or a newer version installed. Then, execute the following command to install the required libraries:

```
pip3 install -r requirements.txt
```



## Source Code Structure 

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

**Please note** that the file names prefixed with 'Fig' and 'Tab' correspond to figures and tables in the paper, respectively. For example, `Tab1_dataset_analyze.py` corresponds to the data in Table 1 of the paper, while `Fig2_spatial_locality.py` corresponds to the results displayed in Figure 2 of the paper.



## Run

As our dataset is currently under internal review, we offer an online platform (refer to Artifact Location) for evaluation. On our platform, utilize the following command to access our source code files:

```
cd ATC_Artifact/
```

If you want to validate our findings and results, please execute the following command:

```
cd <folder>
python3 <filename>.py
```

The results may be directly output in the console, and detailed data can also be obtained from the `Data` folder. Since the prediction model uses machine learning models, the results of Figures 12-15 may vary. Therefore, we also offer the results used in the paper in the `Data\*\result_used_in_paper` folder.



For example, if you want to replicate the results of Table 2 in the paper for error mode analysis, you can execute the following commands:

```
cd Analyses
python3 Fig2_spatial_locality.py
```

Subsequently, you can obtain the following output to verify the results presented in the paper.

```
The results of Table 2 in Sec-3.2

Number of different error modes across various error types
                 All error types   CE  UER  UEO  CE&UER  UEO&UER  CE&UEO&UER
single-cell                  715  555  264    7       0        1           0
two-cell                      67   36   63    2      15        1           0
single-row                     9    5    1    0       4        2           0
row-dominant                  20   17    4    7      16       10           7
two-row                       42   27    4    4      33       11          16
single-column                339  270   90   26      16        8           4
column-dominant               41   16   22   13      11        3           3
two-column                    38    2   10   22       7        7           1
irregular                    159   65   97    6      58        1           1

Percentage of different error modes across various error types (%)
                 All error types        CE       UER       UEO   CE&UER  \
single-cell             0.500000  0.558912  0.475676  0.080460  0.00000
two-cell                0.046853  0.036254  0.113514  0.022989  0.09375
single-row              0.006294  0.005035  0.001802  0.000000  0.02500
row-dominant            0.013986  0.017120  0.007207  0.080460  0.10000
two-row                 0.029371  0.027190  0.007207  0.045977  0.20625
single-column           0.237063  0.271903  0.162162  0.298851  0.10000
column-dominant         0.028671  0.016113  0.039640  0.149425  0.06875
two-column              0.026573  0.002014  0.018018  0.252874  0.04375
irregular               0.111189  0.065458  0.174775  0.068966  0.36250

                  UEO&UER  CE&UEO&UER
single-cell      0.022727     0.00000
two-cell         0.022727     0.00000
single-row       0.045455     0.00000
row-dominant     0.227273     0.21875
two-row          0.250000     0.50000
single-column    0.181818     0.12500
column-dominant  0.068182     0.09375
two-column       0.159091     0.03125
irregular        0.022727     0.03125
```



