import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier


def predict_server_failure(fault_data_label_file, res_file, down_sample=None, pre_threshold=0.6):
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

    rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=50, class_weight="balanced",
                                           criterion="entropy", min_samples_leaf=100, min_samples_split=100)

    rf_classifier.fit(X_train, y_train)

    y_pred_prob = rf_classifier.predict_proba(X_test)
    y_pred = (y_pred_prob[:, 1] > pre_threshold)
    y_pred_rf = rf_classifier.predict(X_test)

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
    print("Results of server-level predictor (Precision, Recall, F1_score)")
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


def predict_bank_failure(fault_data_label_file, res_file, down_sample=None, pre_threshold=0.55):
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

    rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=50, class_weight="balanced",
                                           criterion="entropy",
                                           min_samples_split=10)

    rf_classifier.fit(X_train, y_train)
    y_pred_prob = rf_classifier.predict_proba(X_test)
    y_pred = (y_pred_prob[:, 1] > pre_threshold)
    y_pred_rf = rf_classifier.predict(X_test)

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
    print("Results of bank-level predictor (Precision, Recall, F1_score)")
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


def predict_row_failure(fault_data_label_file, res_file, down_sample=None, pre_threshold=0.55):
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
    rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=50, class_weight="balanced",
                                           criterion="entropy", min_samples_leaf=100, min_samples_split=100)

    rf_classifier.fit(X_train, y_train)  # X_train consists feature，y_train consists labels

    # perform prediction
    y_pred_prob = rf_classifier.predict_proba(X_test)
    y_pred = (y_pred_prob[:, 1] > pre_threshold)  # using a pre-given threshold for prediction
    y_pred_rf = rf_classifier.predict(X_test)

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
    print("Results of row-level predictor (Precision, Recall, F1_score)")
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


def predict_col_failure(fault_data_label_file, res_file, down_sample=None, pre_threshold=0.6):
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

    rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=50, class_weight="balanced",
                                           criterion="entropy")

    rf_classifier.fit(X_train, y_train)

    y_pred_prob = rf_classifier.predict_proba(X_test)
    y_pred = (y_pred_prob[:, 1] > pre_threshold)
    y_pred_rf = rf_classifier.predict(X_test)

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
    print("Results of col-level predictor (Precision, Recall, F1_score)")
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
    input_fold = Path(r"D:\code\HBMErrors\submit_fold\Fig14_diff_observation_window")
    res_fold = input_fold.joinpath("result")
    res_fold.mkdir(exist_ok=True)

    down_sample = 20

    observation_windows = ["1-hour", "1-day", "30-day"]
    for window in observation_windows:
        window_input_fold = input_fold.joinpath(f"{window}_observation_window")

        row_input_file = window_input_fold.joinpath(f"data_for_row-level_prediction.csv")
        col_input_file = window_input_fold.joinpath(f"data_for_col-level_prediction.csv")
        bank_input_file = window_input_fold.joinpath(f"data_for_bank-level_prediction.csv")
        server_input_file = window_input_fold.joinpath(f"data_for_server-level_prediction.csv")

        row_res_file = res_fold.joinpath(f"O{window}_row-level_predictor.csv")
        col_res_file = res_fold.joinpath(f"O{window}_col-level_predictor.csv")
        bank_res_file = res_fold.joinpath(f"O{window}_bank-level_predictor.csv")
        server_res_file = res_fold.joinpath(f"O{window}_server-level_predictor.csv")

        # Number of test
        for i in range(1):
            print(f"=======Test{i + 1} when using {window} observation window=======\n")
            print(row_input_file.absolute())
            predict_row_failure(row_input_file, row_res_file, down_sample=down_sample)
            predict_col_failure(col_input_file, col_res_file, down_sample=down_sample)
            predict_bank_failure(bank_input_file, bank_res_file, down_sample=down_sample)
            predict_server_failure(server_input_file, server_res_file, down_sample=down_sample)

    print(f"The results of above test is saved in {res_fold.absolute()}")
