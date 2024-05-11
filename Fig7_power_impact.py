import time
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

cell_location_header = ["Datacenter", "Server", "Name", "Stack", "SID", "PcId", "BankGroup", "BankArray", "Col", "Row",
                        "Time"]
ip_time_header = cell_location_header[:2] + cell_location_header[-1:]
one_day = 86400  # seconds
a_week = one_day * 7
a_month = one_day * 30


def plot_power_figures(lines_arrays, average_values, save_figures=None):
    plot_header = [["UER average power", "Average power of server"], ["UER peak power", "Peak power of server"]]
    color = [["#1E6B9C", "#278AC9"], ["#2B9F2B", "#53D153"]]
    for i in range(2):
        lines_array = lines_arrays[i]
        average_value = average_values[i]
        n = len(lines_array)

        # x 对应为 (1, N)
        x_values = np.arange(1, n + 1)
        x_values = np.flip(x_values, axis=0)

        # 固定曲线 y=10
        fixed_line = np.full_like(x_values, average_value)

        # 画折线图
        plt.plot(x_values, lines_array, label=plot_header[i][0], linewidth=3)  #
        plt.plot(x_values, fixed_line, label=plot_header[i][1], linestyle='--', linewidth=3)  #
    plt.fill_between(x_values, [2000] * 1008, color='gray', alpha=0.2, where=(x_values >= 820) & (x_values <= 918))
    plt.fill_between(x_values, [2000] * 1008, color='gray', alpha=0.4, where=(x_values >= 918) & (x_values <= 1008))
    plt.plot([918, 918], [1400, 2000], color="black", linestyle='-.', alpha=0.6)
    plt.plot([820, 820], [1400, 2000], color="black", linestyle='-.', alpha=0.6)
    plt.plot([1008, 1008], [1400, 2000], color="black", linestyle='-.', alpha=0.6)
    # plt.plot(x_values, fixed_line, label=plot_header[i][1], linestyle='--')
    # 添加标签和标题

    plt.xlabel(r'(a) Time approaches to UER occurrences', fontsize=27)
    plt.ylabel('Power (W)', fontsize=25)
    # ticks, labels = plt.xticks([0, 288, 576, 864, 1008], ["1 week", "5 days", "3 days", "1 day", "Occur UER"], fontsize=25)
    ticks, labels = plt.xticks([0, 288, 576, 864], ["1 week", "5 days", "3 days", "1 day"],
                               fontsize=25)
    # labels[4].set_color('#851321')
    plt.yticks(fontsize=20)
    plt.ylim(1400, 2000)
    # plt.xlim(-100, 1100)
    # plt.title()
    plt.legend(loc="upper center", bbox_to_anchor=(0.4, 1.0), fontsize=25, ncol=2, frameon=False, labelspacing=0.3,
               handletextpad=0.2, handlelength=1, columnspacing=0.3)
    plt.arrow(915, 1438, 80, 30, head_width=20, head_length=15, width=3, fc='red', ec='red', zorder=10)
    plt.text(850, 1415, "UER", color="red", fontsize=23)
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['right'].set_linewidth(1.5)
    if save_figures:
        fig = plt.gcf()
        fig.set_size_inches(15, 5)
        plt.savefig(save_figures, bbox_inches='tight')
    # 显示图形
    # plt.show()
    print(f"Figure 7(a) is saved in {save_figures.absolute()}")


def plot_ce_power_figures(lines_arrays, average_values, save_figures=None):
    plot_header = [["CE average power", "Average power of server"], ["CE peak power", "Peak power of server"]]
    ce_colors = [["#002060", "#5BA7C4"], ["#9B003B", "#E27985"]]
    for i in range(2):
        lines_array = lines_arrays[i]
        average_value = average_values[i]
        n = len(lines_array)

        # x 对应为 (1, N)
        x_values = np.arange(1, n + 1)
        x_values = np.flip(x_values, axis=0)

        # 固定曲线 y=10
        fixed_line = np.full_like(x_values, average_value)

        # 画折线图

        plt.plot(x_values, lines_array, label=plot_header[i][0], linewidth=3, color=ce_colors[i][0], zorder=4)
        plt.plot(x_values, fixed_line, label=plot_header[i][1], linestyle='--', linewidth=3, color=ce_colors[i][1])

    plt.fill_between(x_values, [2500] * 4320, color='gray', alpha=0.1, where=(x_values >= 2180) & (x_values <= 4320))
    plt.plot([2180, 2180], [1400, 2500], color="black", linestyle='-.', alpha=0.6, zorder=5, linewidth=2)
    plt.plot([4320, 4320], [1400, 2500], color="black", linestyle='-.', linewidth=2, alpha=0.6)

    """
    plt.fill_between(x_values, [2000] * 4320, color='gray', alpha=0.4, where=(x_values >= 918) & (x_values <= 1008))
    plt.plot([918, 918], [1400, 2000], color="black", linestyle='-.',alpha=0.6)
    plt.plot([820, 820], [1400, 2000], color="black", linestyle='-.',alpha=0.6)
    plt.plot([1008, 1008], [1400, 2000], color="black", linestyle='-.',alpha=0.6)
    """
    # plt.plot(x_values, fixed_line, label=plot_header[i][1], linestyle='--')
    # 添加标签和标题
    plt.xlabel(r'(b) Time approaches to CE occurrences', fontsize=27)
    plt.ylabel('Power (W)', fontsize=25)
    plt.xticks([0, 1296, 2304, 3312], ["1 month", "3 weeks", "2 weeks", "1 week"], fontsize=25)
    # ticks,labels = plt.xticks([0, 1296, 2304, 3312, 4320], ["1 month", "3 weeks", "2 weeks", "1 week", "Occur CE"], fontsize=25)
    # labels[4].set_color('#851321')
    plt.yticks(fontsize=20)
    plt.ylim(1400, 2500)
    plt.xlim(-300, 4600)
    plt.tick_params(axis='x', which='both', width=2, length=8)
    # plt.title()
    plt.legend(loc="upper center", bbox_to_anchor=(0.22, 1.0), fontsize=25, ncol=1, frameon=False, labelspacing=0.3,
               handletextpad=0.2, handlelength=1, columnspacing=0.3)
    plt.arrow(4050, 1450, 220, 50, head_width=60, head_length=60, width=5, fc='red', ec='red', zorder=10)
    plt.text(3850, 1415, "CE", color="red", fontsize=23)
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['right'].set_linewidth(1.5)
    if save_figures:
        fig = plt.gcf()
        fig.set_size_inches(15, 5)
        plt.savefig(save_figures, bbox_inches='tight')
    # 显示图形
    # plt.show()
    print(f"Figure 7(b) is saved in {save_figures.absolute()}")


def power_value(tmp_power_data, result_power_number, result_power_value, uer_time, power_type):
    count_number = 0
    for index, row in tmp_power_data.iterrows():
        time_number = int((uer_time - row["Time"]) // 600)

        result_power_number[time_number] += 1
        result_power_value[time_number] += row[power_type]
        count_number += 1


def analyse_power_fluctuate(fault_data, power_fold, consider_time_range=86400, power_type=" Average Power"):
    result_power_number = np.zeros(int(consider_time_range // 600))
    result_power_value = np.zeros(int(consider_time_range // 600))
    sum_mean_peak_power = 0
    sum_mean_peak_number = 0
    for i_index, item in fault_data.groupby(cell_location_header[:2], as_index=False):
        ip_power_file = power_fold.joinpath(i_index[0], f"{i_index[1]}_power.csv")
        if ip_power_file.exists():
            power_data = pd.read_csv(ip_power_file)
            mean_peak_power = power_data[power_type].mean()
            for r_index, row in item.iterrows():
                tmp_power_data = power_data[
                    (power_data["Time"] <= row["Time"]) & (power_data["Time"] > row["Time"] - consider_time_range)]
                if len(tmp_power_data) > 0:
                    power_value(tmp_power_data, result_power_number, result_power_value, row["Time"], power_type)
                    sum_mean_peak_number += 1
                    sum_mean_peak_power += mean_peak_power
    # print(result_power_number, result_power_value)
    power_flow = result_power_value / result_power_number
    power_average = sum_mean_peak_power / sum_mean_peak_number
    return power_flow, power_average


def get_diff_time_power(fault_data, ecc_type, specific_time_range, err_power_fold):
    print(f"Get {err_power_fold} power trend\nThis analysis require a lengthy process")
    power_flows = []
    power_averages = []
    power_types = [" Average Power", " Peak Power"]
    for power_type in power_types:
        power_flow, power_average = analyse_power_fluctuate(
            fault_data[fault_data["EccType"] == ecc_type],
            power_fold=err_power_fold,
            consider_time_range=specific_time_range, power_type=power_type)
        power_flows.append(power_flow)
        power_averages.append(power_average)
    print(f"Done the analysis for {ecc_type}")
    return power_flows, power_averages


if __name__ == "__main__":
    save_fold = Path(r"D:\code\HBMErrors\submit_fold\Fig7_power_impact")
    save_file = save_fold.joinpath(r"power_impact_source.csv")
    power_fold = save_fold.joinpath(r"power")
    res_fold = save_fold.joinpath("result")
    res_fold.mkdir(exist_ok=True)

    source_fault_data = pd.read_csv(save_file)

    uer_power_file = res_fold.joinpath("uer_power.csv")
    ce_power_file = res_fold.joinpath("ce_power.csv")

    if uer_power_file.exists():
        uer_power_flows = np.loadtxt(uer_power_file, delimiter=",")
        uer_power_averages = [1572.243156676282, 1747.2407407497096]
    else:
        # get a week power trend  before UER
        uer_power_flows, uer_power_averages = get_diff_time_power(source_fault_data, "UER", a_week, power_fold)
        np.savetxt(uer_power_file, np.array(uer_power_flows), delimiter=",")
        print(uer_power_averages)

    if ce_power_file.exists():
        ce_power_flows = np.loadtxt(ce_power_file, delimiter=",")
        ce_power_averages = [1556.827847778644, 1752.6175305883414]
    else:
        # get a month power trend before CE
        ce_power_flows, ce_power_averages = get_diff_time_power(source_fault_data, "CE", a_month, power_fold)
        np.savetxt(ce_power_file, np.array(ce_power_flows), delimiter=",")

    print(f"The results of Fig.7 (Sec-3.4)\n")
    print(f"The detailed information of power trend (a week) before UER is saved in {uer_power_file.absolute()}")
    print(f"The detailed information of power trend (a month) before CE is saved in {ce_power_file.absolute()}\n")

    plot_ce_power_figures(ce_power_flows, ce_power_averages, res_fold.joinpath("ce_power.pdf"))
    plot_power_figures(uer_power_flows, uer_power_averages, res_fold.joinpath("uer_power.pdf"))
