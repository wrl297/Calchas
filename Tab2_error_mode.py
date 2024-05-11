from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt

cell_location_header = ["Datacenter", "Server", "Name", "Stack", "SID", "PcId", "BankGroup", "BankArray", "Col", "Row",
                        "Time"]

fault_modes = ["single-cell", "two-cell", "single-row", "row-dominant", "two-row",
               "single-column", "column-dominant", "two-column", "irregular"]

save_fold = Path(r"D:\code\HBMErrors\submit_fold\Tab2_error_mode")
save_file = save_fold.joinpath(r"fault_mode_source.csv")
result_file_fold = save_fold.joinpath(r"result")
result_file_fold.mkdir(exist_ok=True)


def plot_sctter(bank_fault_data, bank_fault_mode, plot_num):
    fig, ax = plt.subplots(figsize=(6, 6))
    ce_data = bank_fault_data[bank_fault_data["EccType"] == "CE"]
    ue_data = bank_fault_data[(bank_fault_data["EccType"] == "UER") | (bank_fault_data["EccType"] == "UEO")]
    plt.scatter(ce_data["Col"], ce_data["Row"], color="#FC7C3B", alpha=0.6, marker="o", s=100)
    plt.scatter(ue_data["Col"], ue_data["Row"], color="#F90000", marker="o", s=100, alpha=0.8)

    # continuous_tick = np.arange(0,33,8)
    plt.xticks([])
    plt.yticks([])
    # plt.xticks(continuous_tick)
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['right'].set_linewidth(1.5)
    plt.xlabel('Column', fontsize=32)
    plt.ylabel('Row', fontsize=32)
    # ax.set_facecolor('#EAEAEA')
    plt.savefig(result_file_fold.joinpath(f'scatter_{bank_fault_mode}_{plot_num}.pdf'))
    # plt.show()


def identify_fault_modes(fault_bank_data, dominant_threshold=0.8):
    fault_bank_data = fault_bank_data.drop_duplicates(subset=cell_location_header[:-1])
    column_header = cell_location_header[:-2]
    row_header = cell_location_header[:-3] + cell_location_header[-2:-1]
    column_num = 0
    column_cell_num = {}
    row_num = 0
    row_cell_num = {}

    for index, item in fault_bank_data.groupby(column_header, as_index=False):
        column_num += 1
        column_cell_num[index[-1]] = len(item)
    for index, item in fault_bank_data.groupby(row_header, as_index=False):
        row_num += 1
        row_cell_num[index[-1]] = len(item)
    # single cell
    if row_num * column_num <= 1:
        return fault_modes[0]
    # two cell
    elif row_num == 2 and column_num == 2 and len(fault_bank_data) == 2:
        return fault_modes[1]
    # single row
    elif row_num == 1 and column_num > 1:
        return fault_modes[2]
    # two row
    elif row_num == 2 and column_num >= 3:
        for key in row_cell_num.keys():
            if row_cell_num[key] <= 1:
                # row dominant
                if 1 - (1 / len(fault_bank_data)) > dominant_threshold:
                    return fault_modes[3]
                else:
                    # irregular
                    return fault_modes[8]
        return fault_modes[4]
    # single column
    elif column_num == 1 and row_num > 1:
        return fault_modes[5]
    # two column
    elif column_num == 2 and row_num >= 3:
        for key in column_cell_num.keys():
            if column_cell_num[key] <= 1:
                # column dominant
                if 1 - (1 / len(fault_bank_data)) > dominant_threshold:
                    return fault_modes[3]
                # irregular
                else:
                    return fault_modes[8]
        return fault_modes[7]
    # others
    else:
        single = 0
        multiple = 0
        for key in row_cell_num.keys():
            if row_cell_num[key] == 1:
                single += 1
            else:
                multiple += row_cell_num[key]
        # row dominant
        if multiple / (multiple + single) > dominant_threshold:
            return fault_modes[3]
        single = 0
        multiple = 0
        for key in column_cell_num.keys():
            if column_cell_num[key] == 1:
                single += 1
            else:
                multiple += column_cell_num[key]
        # column dominant
        if multiple / (multiple + single) > dominant_threshold:
            return fault_modes[6]
        # irregular
        return fault_modes[8]


def analyse_fault_bank(fault_data, useless_types=None, useful_col=1):
    if useless_types:
        for useless_type in useless_types:
            fault_data = fault_data[fault_data["EccType"] != useless_type]
    fault_data["Col"] = fault_data["Col"].apply(lambda x: int(x, 16) // 4)
    fault_data["Row"] = fault_data["Row"].apply(lambda x: int(x, 16))

    bank_header = cell_location_header[:-3]
    fault_mode_count = {}
    for fault_mode in fault_modes:
        fault_mode_count[fault_mode] = 0
    bank_sn = 0
    for index, item in fault_data.groupby(bank_header, as_index=False):
        if useful_col > 1:
            type_nums = len(item.drop_duplicates(subset=["EccType"]))
            if type_nums == useful_col:
                bank_sn += 1
                bank_fault_data = item
                bank_fault_mode = identify_fault_modes(bank_fault_data)
                fault_mode_count[bank_fault_mode] += 1
        else:
            bank_sn += 1
            bank_fault_data = item
            bank_fault_mode = identify_fault_modes(bank_fault_data)
            fault_mode_count[bank_fault_mode] += 1
    return bank_sn, fault_mode_count


if __name__ == "__main__":
    raw_fault_data = pd.read_csv(save_file)

    choice = [[["None"], 1, "All error types"], [["UEO", "UER"], 1, "CE"], [["CE", "UEO"], 1, "UER"],
              [["UER", "CE"], 1, "UEO"],
              [["UEO"], 2, "CE&UER"], [["CE"], 2, "UEO&UER"],  # [["UER"], 2],
              [["None"], 3, "CE&UEO&UER"]]
    all_mode_count = {}
    each_choice_bank_num = {}
    for item in choice:
        bank_num, fault_mode_count = analyse_fault_bank(raw_fault_data, useless_types=item[0], useful_col=item[1])
        all_mode_count[item[2]] = fault_mode_count
        each_choice_bank_num[item[2]] = bank_num
    # output the results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    fault_mode_count_df = pd.DataFrame.from_dict(all_mode_count, orient='index')
    bank_count_se = pd.Series(each_choice_bank_num)

    # fault_mode_count_df = fault_mode_count_df.T
    fault_mode_percentage_df = fault_mode_count_df.div(bank_count_se, axis=0)
    print("\nThe results of Table 2 in Sec-3.2")
    print("\nNumber of different error modes across various error types")
    print(fault_mode_count_df.T)
    print("\nPercentage of different error modes across various error types (%)")
    print(fault_mode_percentage_df.T)
