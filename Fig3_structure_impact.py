from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

cell_location_header = ["Datacenter", "Server", "Name", "Stack", "SID", "PcId", "BankGroup", "BankArray", "Col", "Row","Time"]

number_of_level = {
    cell_location_header[2]: 8, cell_location_header[3]: 4,
    cell_location_header[4]: 2, cell_location_header[5]: 16,
    cell_location_header[6]: 4, cell_location_header[7]: 4,
    cell_location_header[8]: 128, cell_location_header[9]: 16384
}


def plot_diff_levels(data, result_file_fold):
    fig, ax = plt.subplots()

    markers = ['o', 's', '^', 'D', '>', '<', 'p']
    colors = ["#000099", "#FDAE62", "#9B003B", "#D53E4F", "#FAAB8D", "#9CC9DB", "#8984BE"]

    for i, (line_name, y_values) in enumerate(data.items()):
        x_values = np.arange(len(y_values))
        ax.plot(x_values, y_values, marker=markers[i], label=line_name, markersize=8, linewidth=2, color=colors[i])

    ax.legend(loc="upper center", bbox_to_anchor=(0.56, 1.02), fontsize=23, ncol=3, labelspacing=0.4,
              handletextpad=0.02, frameon=False, columnspacing=0.5)

    rect = Rectangle((-1, 400), 3, 120, linewidth=2, edgecolor='red', facecolor='none', linestyle="dashed")
    ax.add_patch(rect)
    ax.xaxis.set_label_coords(0.38, -0.1)
    key_conclusions = {
        'Col_1': (6, 175, 2, 50),
        'Col_2': (14, 265, 2, 50),
        'Col_3': (22, 230, 2, 50),
        'Col_4': (30, 370, 2, 50),
    }

    for line_name, coords in key_conclusions.items():
        rect = Rectangle((coords[0], coords[1]), coords[2], coords[3], linewidth=2, edgecolor="#3B386A",
                         linestyle="dashed",
                         facecolor='none')
        ax.add_patch(rect)

    ax.set_xlabel('(a) Component positions', fontsize=32)
    ax.set_ylabel('Number of components \nwith errors', fontsize=32)
    plt.xticks(np.arange(0, max(map(len, data.values())) + 1, 8), fontsize=25)
    plt.yticks(np.arange(0, 500, 150), fontsize=22)

    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['right'].set_linewidth(1.5)

    fig.set_size_inches(7, 6.5)

    fig = plt.gcf()
    save_figures = result_file_fold.joinpath("fig3a_structure_analysis.pdf")
    plt.savefig(save_figures, bbox_inches='tight')
    #plt.show()
    print(f"\nFigure 3(a) is saved in {save_figures.absolute()}")


def plot_die(die_number, save_fold=None):
    die_data = [np.insert(die_number, 4, 0)]

    plot_info = [[["#CA2421", "#F8AB8D", "#99C9DB", "#287AB2", "black", "#CA2421", "#F8AB8D", "#99C9DB", "#287AB2"],
                  ["", "", "", "", "/", "/", "/", "/"],
                  '(b) Number of dies with errors',
                  "die_location_diff"]]
    x_labels = []
    for i in range(9):
        if i == 4:
            x_labels.append(f" ")
        elif i < 4:
            x_labels.append(f"{7 - i}")
        else:
            x_labels.append(f"{8 - i}")
    lim_value = [20, 30]
    loc_value = [10, 25]
    for i in range(1):
        fig, ax = plt.subplots(figsize=(6, 6.3))

        tmp_data = die_data[i]
        tmp_data = np.flip(tmp_data)
        continuous_number = np.arange(len(tmp_data))
        # continuous_number = np.flip(continuous_number)
        rect_1 = Rectangle((0, -1), tmp_data[0] + lim_value[i], 5, linewidth=2, edgecolor='none', facecolor='#EAEAEA')
        rect_2 = Rectangle((0, 4), tmp_data[0] + lim_value[i], 5, linewidth=2, edgecolor='none', facecolor='#F4F4F4',
                           alpha=0.6)
        ax.add_patch(rect_1)
        ax.add_patch(rect_2)
        plt.barh(continuous_number, tmp_data, color=plot_info[i][0], edgecolor="black", linewidth=1.5)

        plt.xlabel(plot_info[i][2], fontsize=26)
        plt.ylabel('Positions of dies', fontsize=26)
        plt.xticks(np.arange(0, tmp_data[0] + lim_value[i], 40), fontsize=20)
        plt.subplots_adjust(bottom=0.23)
        ax.xaxis.set_label_coords(0.43, -0.1)

        plt.xlim(0, tmp_data[0] + lim_value[i])
        plt.ylim(-1, 9)
        plt.plot([0, tmp_data[0] + lim_value[i]], [4, 4], linestyle="--", color="grey")
        plt.text(tmp_data[0] - loc_value[i] - 5, 8, "SID 0", fontsize=24)
        plt.text(tmp_data[0] - loc_value[i] - 5, 3, "SID 1", fontsize=24)

        plt.yticks(continuous_number, x_labels, fontsize=22)
        plt.gca().spines['top'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['right'].set_linewidth(1.5)
        if save_fold:
            save_figure = save_fold.joinpath(f"fig3b_{plot_info[i][3]}.pdf")
            plt.savefig(save_figure)

    print(f"Figure 3(b) is saved in {save_figure.absolute()}")


def remap_pcid(pcid_value):
    return hex(int(pcid_value, base=16) // 4)


def unique_hbm_location(fault_level_data, level_num, is_die=False, is_npu=False):
    count_num = []
    fault_level_data = fault_level_data.drop_duplicates()
    if is_die:
        fault_level_data["PcId"] = fault_level_data["PcId"].map(lambda x: remap_pcid(x))
        fault_level_data = fault_level_data.drop_duplicates()
        for i in range(2):
            for j in range(4):
                tmp_data = fault_level_data[(fault_level_data["SID"] == hex(i)) & (fault_level_data["PcId"] == hex(j))]
                count_num.append(len(tmp_data))
    elif is_npu:
        for i in range(number_of_level[cell_location_header[level_num]]):
            tmp_data = fault_level_data[(fault_level_data[cell_location_header[level_num]] == f"NPU{i + 1}")]
            count_num.append(len(tmp_data))
    else:
        for i in range(number_of_level[cell_location_header[level_num]]):
            tmp_data = fault_level_data[(fault_level_data[cell_location_header[level_num]] == hex(i))]
            count_num.append(len(tmp_data))
    return count_num


def diff_level_distribution(fault_data):
    all_count_num = {}
    for i in range(2, 9):
        is_npu = False
        # 统计bank
        level_fault_data = fault_data[cell_location_header[:i + 1]]
        if i == 2:
            is_npu = True
        specific_count_num = unique_hbm_location(level_fault_data, i, is_npu=is_npu)
        if i == 8:
            specific_count_num = specific_count_num[0::4]
        # ., np.sum(specific_count_num))
        all_count_num[cell_location_header[i]] = specific_count_num
    specific_count_num = unique_hbm_location(fault_data[cell_location_header[:6]], level_num=6, is_die=True)
    all_count_num["Die"] = specific_count_num
    return all_count_num


if __name__ == "__main__":
    save_fold = Path(r"D:\code\HBMErrors\final_used")
    save_file = save_fold.joinpath(r"structure_impcat_source.csv")
    result_file_fold = save_fold.joinpath(r"result")
    result_file_fold.mkdir(exist_ok=True)

    raw_fault_data = pd.read_csv(save_file)

    # count # of levels with error in specific position
    all_count_num = diff_level_distribution(raw_fault_data)
    fig_header = ["DSA", "HBM", "SID", "PS-CH", "BG", "Bank", "Col", "Die"]
    # output results
    print("The results of Figure 3 in Sec-3.2")
    print("\n# of components in specific positions (1-N)")
    key_header = list(all_count_num.keys())
    for header_index in range(len(fig_header)):
        print(
            f"{fig_header[header_index]}(1-{len(all_count_num[key_header[header_index]])}): {all_count_num[key_header[header_index]]}")
        all_count_num[fig_header[header_index]] = all_count_num.pop(key_header[header_index])

    # plot figs
    die_num = all_count_num.pop("Die")
    plot_diff_levels(all_count_num, result_file_fold)
    plot_die(die_num, result_file_fold)
