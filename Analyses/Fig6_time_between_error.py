from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

cell_location_header = ["Datacenter", "Server", "Name", "Stack", "SID", "PcId", "BankGroup", "BankArray", "Col", "Row",
                        "Time"]

means_dict = {
    "CE_f": "First CE to UER",
    "CE_l": "Last CE to UER",
    "UER_2": "Two successive UERs"
}


def plot_hist_figures(time_distributions, save_file=None):
    fig, ax = plt.subplots(figsize=(17, 6.5))
    width = 0.3
    figures_para = {
        "CE_f": ["\u25B3$T_{CE_{first}\\rightarrow UER}$", "#E0EEEE", -1 * width, "--", "o", "#5FA7A6", "x"],
        "CE_l": ["\u25B3$T_{CE_{last}\\rightarrow UER}$", "#009BCE", 0, "-.", "v", "#0098CA", ""],
        "UER_2": ["\u25B3$T_{UER\\rightarrow UER_{next}}$", "#8B0000", width, "-", "d", "#8B0000", ""]
    }
    np.set_printoptions(precision=3)
    print("\nThe Percentage of three time interval (Fig.6 in Sec-3.3)")
    all_percentage = {}
    for key in time_distributions.keys():
        if key in ["UEO", "UER_r"]:
            continue
        time_data = time_distributions[key]
        number_use_time = len(time_data)
        time_data = np.array(time_data)
        time_intervals = [0, 3600, 86400, 2592000, 31104000]  # 60, 600,
        time_intervals_labels = ["0", "1 h", "1 d", "30 d", "365 d", "inf"]  # "1 min", "10 min",
        counts_interval = []
        interval_ticks = []
        cumulative_values = []
        for i in range(len(time_intervals)):
            if i < len(time_intervals) - 1:
                counts_interval.append(np.sum((time_data >= time_intervals[i]) & (time_data < time_intervals[i + 1])))
                interval_ticks.append(f"[{time_intervals_labels[i]}, {time_intervals_labels[i + 1]})")
            else:
                counts_interval.append(np.sum(time_data >= time_intervals[i]))
                interval_ticks.append(f"[{time_intervals_labels[i]}, {time_intervals_labels[i + 1]})")

        counts_interval = np.array(counts_interval) / number_use_time

        for i in range(len(counts_interval)):
            if i == 0:
                cumulative_values.append(counts_interval[i])
            else:
                cumulative_values.append(counts_interval[i] + cumulative_values[i - 1])
        if key == "CE_f":
            all_percentage["Interval"] = interval_ticks
        all_percentage[means_dict[key]] = counts_interval
        # print(f"{means_dict[key]}\t", counts_interval)
        # counts_interval = counts_interval# * 10 / 9

        continuous_number = np.arange(0, len(counts_interval))

        bars = ax.bar(continuous_number + figures_para[key][2], counts_interval, width=width,
                      label=figures_para[key][0],
                      align='center', alpha=0.7, color=figures_para[key][1],
                      edgecolor='black', hatch=figures_para[key][6])

        # for bar in bars:
        #    yval = bar.get_height()
        #    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval * 100:.1f}'.format(yval), ha='center', va='bottom',
        #                 fontsize=22)
        ax.plot(continuous_number, cumulative_values, marker=figures_para[key][4], linestyle=figures_para[key][3],
                color=figures_para[key][5], label=r" ", linewidth=3, markersize=9)

    all_percentage_df = pd.DataFrame.from_dict(all_percentage)
    print(all_percentage_df)

    x_ticks = continuous_number
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(interval_ticks, fontsize=25)
    # ax.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax_ticks = np.linspace(0, 1, 5)  # 第二个 Y 轴的标签
    ax_labels = [f'{tick * 100:.0f}' for tick in ax_ticks]  # 转换为百分比字符串
    ax.set_yticks(ax_ticks)
    ax.set_yticklabels(ax_labels, fontsize=22)
    ax.set_ylabel("Percentage of \neach interval (%)", fontsize=25)

    ax2_ticks = np.linspace(0, 1, 5)  # 第二个 Y 轴的标签
    ax2_labels = [f'{tick * 100:.0f}' for tick in ax2_ticks]  # 转换为百分比字符串

    ax2 = ax.twinx()
    ax2.set_yticks(ax2_ticks)
    ax2.set_yticklabels(ax2_labels, fontsize=22)
    ax2.set_ylabel('Cumulative probability (%)', color='black', fontsize=25)
    ax.set_ylim(0, 1.1)
    ax2.set_ylim(0, 1.1)
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 4, 2, 5, 3, 6]
    order = [1, 2, 3, 4, 5, 6]
    ax.legend([handles[idx - 1] for idx in order], [labels[idx - 1] for idx in order], loc="upper center",
              bbox_to_anchor=(0.82, 0.7), ncol=2, frameon=False, labelspacing=0.3,
              handletextpad=0.2, handlelength=1, columnspacing=0.3, fontsize=26)

    ax.set_xlabel('Time intervals', fontsize=30)
    plt.subplots_adjust(bottom=0.2)
    if save_file:
        plt.savefig(save_file)
    # plt.show()
    print(f"Figure 6 is saved in {save_file.absolute()}")


def interval_with_other(uer_time, error_type, fault_bank_data, is_first=False):
    interval = []
    type_bank_data = fault_bank_data[fault_bank_data["EccType"] == error_type]
    type_bank_data = type_bank_data[type_bank_data["Time"] <= uer_time]
    if len(type_bank_data) >= 1:
        if is_first:
            first_error_time = type_bank_data["Time"].min()
            interval.append(uer_time - first_error_time)
        else:
            last_error_time = type_bank_data["Time"].max()
            interval.append(uer_time - last_error_time)
    return interval


def time_interval(fault_bank_data, time_distribution):
    uer_items = fault_bank_data[fault_bank_data["EccType"] == "UER"]
    uer_items = uer_items.sort_values(by=["Time"])
    if len(uer_items) > 1:
        time_first = 0
        time_before = 0
        for index, row in uer_items.iterrows():
            time_value = row["Time"]
            time_distribution["CE_f"] += (interval_with_other(time_value, "CE", fault_bank_data, is_first=True))
            time_distribution["CE_l"] += (interval_with_other(time_value, "CE", fault_bank_data))
            if time_first == 0:
                time_first = time_value
            if time_before != 0:
                time_distribution["UER_2"].append(time_value - time_before)
            time_before = time_value
        time_distribution["UER_r"].append(time_value - time_first)
    else:
        for index, row in uer_items.iterrows():
            time_value = row["Time"]
            time_distribution["CE_f"] += (interval_with_other(time_value, "CE", fault_bank_data, is_first=True))
            time_distribution["CE_l"] += (interval_with_other(time_value, "CE", fault_bank_data))
            time_distribution["UEO"] += (interval_with_other(time_value, "UEO", fault_bank_data))


def analyse_bank_data(fault_data, group_header=cell_location_header[:8]):
    time_distribution = {"CE_f": [], "CE_l": [], "UER_2": [], "UEO": [], "UER_r": []}
    for index, item in fault_data.groupby(group_header, as_index=False):
        if len(item) > 1 and len(item[item["EccType"] == "UER"]) >= 1:
            time_interval(item, time_distribution)
    return time_distribution


if __name__ == "__main__":
    save_fold = Path(r"D:\code\HBMErrors\final_used")
    save_file = save_fold.joinpath(r"time_between_error_source.csv")
    result_file_fold = save_fold.joinpath(r"result")
    result_file_fold.mkdir(exist_ok=True)

    source_fault_data = pd.read_csv(save_file)
    source_fault_data = source_fault_data[source_fault_data["Time"] > 1600000000]
    source_fault_data.drop_duplicates(inplace=True)
    source_fault_data.to_csv(save_file, index=None)

    time_distributions = analyse_bank_data(source_fault_data)
    time_distributions_file = result_file_fold.joinpath("time_between_errors.txt")
    tdf = open(time_distributions_file, "w")

    for k, v in time_distributions.items():
        if k in ["UEO", "UER_r"]:
            continue
        tdf.write(f"{means_dict[k]}:{str(v)[1:-1]}\n\n")
    tdf.close()
    print(f"The detailed results of three time distribution is saved in {time_distributions_file.absolute()}")

    # out
    plot_hist_figures(time_distributions, result_file_fold.joinpath("time_distribution.pdf"))
