import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


def predict_server_failure(fault_data_label_file, res_file, down_sample=None, pre_threshold=0.8, model_name="RF"):
    data = np.loadtxt(fault_data_label_file, delimiter=",", skiprows=1, dtype="str")
    data = data.astype("float")

    X = data[:, :-1]
    y = data[:, -1]

    if down_sample:
        neagative_sample = np.where(y == 0)[0]
        positive_sample = np.where(y == 1)[0]
        neagtive_number = len(positive_sample) * down_sample
        if neagtive_number < len(neagative_sample):
            random_selection = np.random.choice(neagative_sample, size=neagtive_number, replace=False)
            sample_index = np.r_[random_selection, positive_sample]
            sample_index.sort()
            X = X[sample_index]
            y = y[sample_index]

    data_items_num = len(X)
    train_num = int(data_items_num * 0.7)

    X_train = X[:train_num]
    X_test = X[train_num:]
    y_train = y[:train_num]
    y_test = y[train_num:]

    is_prob = True
    if model_name == "GBDT":
        rf_classifier = GradientBoostingClassifier(n_estimators=200, max_depth=50, learning_rate=30,
                                                   min_samples_leaf=100, min_samples_split=100)
    elif model_name == "SVM":
        rf_classifier = SVC(kernel='linear', C=1, gamma='scale', max_iter=50)
        is_prob = False
    else:
        rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=50, class_weight="balanced",
                                               criterion="entropy", min_samples_leaf=100, min_samples_split=100)

    rf_classifier.fit(X_train, y_train)

    y_pred_rf = rf_classifier.predict(X_test)
    if is_prob:
        y_pred_prob = rf_classifier.predict_proba(X_test)
        y_pred = (y_pred_prob[:, 1] > pre_threshold)
    else:
        y_pred = y_pred_rf.astype("bool")

    y_test_bool = y_test.astype("bool")

    tp = np.sum(y_pred & y_test_bool)
    fp = np.sum(y_pred & (~y_test_bool))
    tn = np.sum((~y_pred) & (~y_test_bool))
    fn = np.sum((~y_pred) & y_test_bool)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    res_file_item = open(res_file, "a")
    res_file_item.write(f"prob-{pre_threshold}:, {accuracy}, {precision}, {recall}, {f1_score}\n")
    print(f"Results of server-level predictor based on {model_name} (Precision, Recall, F1_score)")
    print(f"RF with threshold={pre_threshold}: {precision}, {recall}, {f1_score}")

    y_pred_rf = y_pred_rf.astype("bool")
    tp = np.sum(y_pred_rf & y_test_bool)
    fp = np.sum(y_pred_rf & (~y_test_bool))
    tn = np.sum((~y_pred_rf) & (~y_test_bool))
    fn = np.sum((~y_pred_rf) & y_test_bool)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    res_file_item = open(res_file, "a")
    res_file_item.write(f"rf:, {accuracy}, {precision}, {recall}, {f1_score}\n")
    print(f"Default RF: {precision}, {recall}, {f1_score}\n")
    res_file_item.close()


def predict_bank_failure(fault_data_label_file, res_file, down_sample=None, pre_threshold=0.95, model_name="RF"):
    data = np.loadtxt(fault_data_label_file, delimiter=",", skiprows=1, dtype="str")
    data = data.astype("float")

    X = data[:, :-1]
    y = data[:, -1]

    if down_sample:
        neagative_sample = np.where(y == 0)[0]
        positive_sample = np.where(y == 1)[0]
        neagtive_number = len(positive_sample) * down_sample
        if neagtive_number < len(neagative_sample):
            random_selection = np.random.choice(neagative_sample, size=neagtive_number, replace=False)
            sample_index = np.r_[random_selection, positive_sample]
            sample_index.sort()
            X = X[sample_index]
            y = y[sample_index]

    data_items_num = len(X)
    train_num = int(data_items_num * 0.7)

    X_train = X[:train_num]
    X_test = X[train_num:]
    y_train = y[:train_num]
    y_test = y[train_num:]

    is_prob = True
    if model_name == "GBDT":
        rf_classifier = GradientBoostingClassifier(n_estimators=1000, max_depth=50, learning_rate=0.1,
                                                   min_samples_split=10)  # ,
    elif model_name == "SVM":
        rf_classifier = SVC(kernel='rbf', C=60, gamma='scale', max_iter=70)
        is_prob = False
    else:
        rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=50, class_weight="balanced",
                                               criterion="entropy", min_samples_leaf=100, min_samples_split=100)

    rf_classifier.fit(X_train, y_train)

    y_pred_rf = rf_classifier.predict(X_test)
    if is_prob:
        y_pred_prob = rf_classifier.predict_proba(X_test)
        y_pred = (y_pred_prob[:, 1] > pre_threshold)
    else:
        y_pred = y_pred_rf.astype("bool")

    y_test_bool = y_test.astype("bool")

    tp = np.sum(y_pred & y_test_bool)
    fp = np.sum(y_pred & (~y_test_bool))
    tn = np.sum((~y_pred) & (~y_test_bool))
    fn = np.sum((~y_pred) & y_test_bool)
    res_file_item = open(res_file, "a")

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    print(f"Results of bank-level predictor based on {model_name} (Precision, Recall, F1_score)")
    print(f"RF with threshold={pre_threshold}: {precision}, {recall}, {f1_score}")
    res_file_item.write(f"prob-{pre_threshold}: {accuracy}, {precision}, {recall}, {f1_score}\n")

    y_pred_rf = y_pred_rf.astype("bool")
    tp = np.sum(y_pred_rf & y_test_bool)
    fp = np.sum(y_pred_rf & (~y_test_bool))
    tn = np.sum((~y_pred_rf) & (~y_test_bool))
    fn = np.sum((~y_pred_rf) & y_test_bool)
    print(f"Default RF: {precision}, {recall}, {f1_score}\n")
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    res_file_item.write(f"rf:, {accuracy}, {precision}, {recall}, {f1_score}\n")
    res_file_item.close()


def predict_row_failure(fault_data_label_file, res_file, down_sample=None, pre_threshold=0.8, model_name="RF"):
    data = np.loadtxt(fault_data_label_file, delimiter=",", skiprows=1, dtype="str")
    data = data.astype("float")
    X = data[:, :-1]
    y = data[:, -1]
    # perform down sample using a pre-given ratio
    if down_sample:
        neagative_sample = np.where(y == 0)[0]
        positive_sample = np.where(y == 1)[0]
        neagtive_number = len(positive_sample) * down_sample
        if neagtive_number < len(neagative_sample):
            random_selection = np.random.choice(neagative_sample, size=neagtive_number, replace=False)
            sample_index = np.r_[random_selection, positive_sample]
            sample_index.sort()
            X = X[sample_index]
            y = y[sample_index]

    data_items_num = len(X)
    train_num = int(data_items_num * 0.7)

    # split data
    X_train = X[:train_num]
    X_test = X[train_num:]
    y_train = y[:train_num]
    y_test = y[train_num:]

    # build model
    is_prob = True
    if model_name == "GBDT":
        rf_classifier = GradientBoostingClassifier(n_estimators=50, max_depth=30, learning_rate=0.6)  # ,
    elif model_name == "SVM":
        rf_classifier = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced')
        is_prob = False
    else:
        rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=50, class_weight="balanced",
                                               criterion="entropy", min_samples_leaf=100, min_samples_split=100)

    rf_classifier.fit(X_train, y_train)

    y_pred_rf = rf_classifier.predict(X_test)
    if is_prob:
        y_pred_prob = rf_classifier.predict_proba(X_test)
        y_pred = (y_pred_prob[:, 1] > pre_threshold)
    else:
        y_pred = y_pred_rf.astype("bool")

    y_test_bool = y_test.astype("bool")

    tp = np.sum(y_pred & y_test_bool)
    fp = np.sum(y_pred & (~y_test_bool))
    tn = np.sum((~y_pred) & (~y_test_bool))
    fn = np.sum((~y_pred) & y_test_bool)
    res_file_item = open(res_file, "a")
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    print(f"Results of row-level predictor based on {model_name} (Precision, Recall, F1_score)")
    print(f"RF with threshold={pre_threshold}: {precision}, {recall}, {f1_score}")
    res_file_item.write(f"prob-{pre_threshold}:, {accuracy}, {precision}, {recall}, {f1_score}\n")

    # using the default threshold (0.5) for prediction
    y_pred_rf = y_pred_rf.astype("bool")
    tp = np.sum(y_pred_rf & y_test_bool)
    fp = np.sum(y_pred_rf & (~y_test_bool))
    tn = np.sum((~y_pred_rf) & (~y_test_bool))
    fn = np.sum((~y_pred_rf) & y_test_bool)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    print(f"Default RF: {precision}, {recall}, {f1_score}\n")
    res_file_item.write(f"Default RF:, {accuracy}, {precision}, {recall}, {f1_score}\n")
    res_file_item.close()


def predict_col_failure(fault_data_label_file, res_file, down_sample=None, pre_threshold=0.7, model_name="RF"):
    data = np.loadtxt(fault_data_label_file, delimiter=",", skiprows=1, dtype="str")
    data = data.astype("float")
    X = data[:, :-1]
    y = data[:, -1]

    if down_sample:
        neagative_sample = np.where(y == 0)[0]
        positive_sample = np.where(y == 1)[0]
        neagtive_number = len(positive_sample) * down_sample
        if neagtive_number < len(neagative_sample):
            random_selection = np.random.choice(neagative_sample, size=neagtive_number, replace=False)
            sample_index = np.r_[random_selection, positive_sample]
            sample_index.sort()
            X = X[sample_index]
            y = y[sample_index]

    data_items_num = len(X)
    train_num = int(data_items_num * 0.7)

    X_train = X[:train_num]
    X_test = X[train_num:]
    y_train = y[:train_num]
    y_test = y[train_num:]

    is_prob = True
    if model_name == "GBDT":
        rf_classifier = GradientBoostingClassifier(n_estimators=200, max_depth=50, learning_rate=0.2)  # ,
    elif model_name == "SVM":
        rf_classifier = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', decision_function_shape='ovo')
        is_prob = False
    else:
        rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=50, class_weight="balanced",
                                               criterion="entropy", min_samples_leaf=100, min_samples_split=100)

    rf_classifier.fit(X_train, y_train)

    y_pred_rf = rf_classifier.predict(X_test)
    if is_prob:
        y_pred_prob = rf_classifier.predict_proba(X_test)
        y_pred = (y_pred_prob[:, 1] > pre_threshold)
    else:
        y_pred = y_pred_rf.astype("bool")

    y_test_bool = y_test.astype("bool")

    tp = np.sum(y_pred & y_test_bool)
    fp = np.sum(y_pred & (~y_test_bool))
    tn = np.sum((~y_pred) & (~y_test_bool))
    fn = np.sum((~y_pred) & y_test_bool)
    res_file_item = open(res_file, "a")
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    print(f"Results of col-level predictor based on {model_name} (Precision, Recall, F1_score)")
    print(f"RF with threshold={pre_threshold}:, {precision}, {recall}, {f1_score}")
    res_file_item.write(f"prob-{pre_threshold}:, {accuracy}, {precision}, {recall}, {f1_score}\n")
    y_pred_rf = y_pred_rf.astype("bool")
    tp = np.sum(y_pred_rf & y_test_bool)
    fp = np.sum(y_pred_rf & (~y_test_bool))
    tn = np.sum((~y_pred_rf) & (~y_test_bool))
    fn = np.sum((~y_pred_rf) & y_test_bool)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    print(f"Default RF:, {precision}, {recall}, {f1_score}\n")
    res_file_item.write(f"rf:, {accuracy}, {precision}, {recall}, {f1_score}\n")
    res_file_item.close()


if __name__ == "__main__":
    input_fold = Path(r"D:\code\HBMErrors\submit_fold\Fig13_diff_model")
    res_fold = input_fold.joinpath("result")
    res_fold.mkdir(exist_ok=True)

    row_input_file = input_fold.joinpath(f"data_for_row-level_prediction.csv")
    col_input_file = input_fold.joinpath(f"data_for_col-level_prediction.csv")
    bank_input_file = input_fold.joinpath(f"data_for_bank-level_prediction.csv")
    server_input_file = input_fold.joinpath(f"data_for_server-level_prediction.csv")

    model_names = ["SVM", "GBDT"]  # "RF" is tested in Fig12_prediction_performance.py
    down_sample = 20

    # Number of test
    for i in range(1):
        print(f"=======Test{i+1} for each predictor=======\n")
        for employ_model in model_names:
            row_res_file = res_fold.joinpath(f"{employ_model}_row-level_predictor.csv")
            col_res_file = res_fold.joinpath(f"{employ_model}_col-level_predictor.csv")
            bank_res_file = res_fold.joinpath(f"{employ_model}_bank-level_predictor.csv")
            server_res_file = res_fold.joinpath(f"{employ_model}_server-level_predictor.csv")

            predict_row_failure(row_input_file, row_res_file, down_sample=down_sample, model_name=employ_model)
            predict_col_failure(col_input_file, col_res_file, down_sample=down_sample, model_name=employ_model)
            predict_bank_failure(bank_input_file, bank_res_file, down_sample=down_sample, model_name=employ_model)
            predict_server_failure(server_input_file, server_res_file, down_sample=down_sample, model_name=employ_model)
    print(f"The results of above test is saved in {res_fold.absolute()}")
