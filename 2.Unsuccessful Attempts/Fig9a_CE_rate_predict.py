import pandas as pd
path= "data/one_hour_predict.csv"
# columns_to_keep = ['IP', 'LogNum', 'Time', 'EccType', 'Position', 'Name', 'Stack', 'PcId', 'SID', 'BankArray', 'BankGroup', 'Row', 'Col', 'Bank_level_1d_DeltCE', 'Bank_label']
df = pd.read_csv(path)
threshold = [500,1000]
all_positive = len(df[df["Bank_label"] == 1])
print("=========================CE Rate Indicator=========================")
for value in threshold:
    predict_positive=len(df[(df["Bank_level_1d_DeltCE"] > value)])
    tp=len(df[(df["Bank_level_1d_DeltCE"] > value) & (df["Bank_label"] == 1)])
    precision = tp/predict_positive
    recall = tp/all_positive
    F1_score = 2*precision*recall/(precision+recall)
    print("*****************************************************************")
    print(f"=============Threshold {value}=============")
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", F1_score)
