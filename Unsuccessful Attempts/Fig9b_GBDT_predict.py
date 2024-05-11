from sklearn.ensemble import GradientBoostingClassifier  # 导入GBDT分类器
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
import pandas as pd
import numpy as np


path = "data/1d_bank_predict.csv"
df = pd.read_csv(path)
# index
total_rows = len(df)
split_ratio = 0.7
split_index = int(total_rows * split_ratio)

# split data
train_data = df[:split_index]
test_data = df[split_index:]

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]
gbdt = GradientBoostingClassifier(n_estimators=100, max_depth=7,
                                  min_samples_leaf=10, min_samples_split=10)
gbdt.fit(X_train, y_train)
y_pred_prob = gbdt.predict_proba(X_test)[:, 1]

threshold = 0.05
y_pred_custom = (y_pred_prob > threshold).astype(int)
confusion_custom = confusion_matrix(y_test, y_pred_custom)
TP_custom = confusion_custom[1, 1]
TN_custom = confusion_custom[0, 0]
FP_custom = confusion_custom[0, 1]
FN_custom = confusion_custom[1, 0]
# performance
recall = recall_score(y_test, y_pred_custom)
precision = precision_score(y_test, y_pred_custom)
F1_score = 2*precision*recall/(precision+recall)
print("=======================GBDT Predictor=======================")
print('Precision:', precision)
print(' Recall :', recall)
print("F1 Score:",F1_score)