from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from sklearn.metrics import recall_score, precision_score

df = pd.read_csv("data/1d_bank_predict.csv")
# df.drop_duplicates(subset=)
total_rows = len(df)
split_ratio = 0.7
split_index = int(total_rows * split_ratio)
# Split Data
train_data = df[:split_index]  # Use the first 70% as the training set.
test_data = df[split_index:]   # Use the last 30% as the test set.
X_train = train_data.iloc[:, :-1]  # Train Features
y_train = train_data.iloc[:, -1]   # Train Labels
X_test = test_data.iloc[:, :-1]    # Test Features
y_test = test_data.iloc[:, -1]     # Test Labels
forest = RandomForestClassifier(n_estimators=1000, max_depth=7, class_weight="balanced",
                                           criterion="entropy", min_samples_leaf=10,min_samples_split=10)  #
forest.fit(X_train, y_train)
y_pred=forest.predict(X_train)
y_proba = forest.predict_proba(X_test)
threshold = 0.25
y_test_pred = (y_proba[:, 1] > threshold).astype(int)
turetp=np.where(y_test>0)
# print(turetp)
confusion=confusion_matrix(y_test, y_test_pred)
TP = confusion[1, 1]
print(len(y_train [y_train ==1]))
print(len(y_test[y_test==1]))
# print("TP",TP)
TN = confusion[0, 0]
# print("TN",TN)
FP = confusion[0, 1]
# print("FP",FP)
FN = confusion[1, 0]
# print("FN",FN)
# print("TP+FP",(TP+FP))
# print("TP + FN",(TP + FN))
Precision=TP / float(TP + FP)
Recall=TP / float(TP + FN)
F1_score=2*Precision*Recall/(Precision+Recall)
print("=======================Random Forest Predictor=======================")
print('Precision:',Precision)
print('Recall:',Recall)
print("F1-score",F1_score)



