from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

spatial_header = ["Datacenter", "Server", "Name", "Stack", "SID", "PcId", "BankGroup", "BankArray", "Col", "Row"]


def plot_figures_merge(level_data, cell_data, weight_name, header, y_label="Percentage (%)", use_color=["blue", "red"],
                       save_figures=None):
    data = [[], []]
    bar_patters = ["", "\\"]
    for i in range(len(level_data)):
        data[1].append(level_data[i][1] / level_data[i][0] * 100)
        data[0].append(cell_data[i][1] / cell_data[i][0] * 100)
    data = np.array(data)
    np.set_printoptions(precision=3)
    print("Percentage of errors in multiple cells")
    print(f"Device level:\t\t {str(fig_header)[1:-1]}")
    print(f"# of error cells:\t {str(np.array(cell_data)[:,0])[1:-1]}\n# of multi-cells:\t {str(np.array(cell_data)[:,1])[1:-1]}")
    print(f"Percentage(%):\t\t {str(data[0])[1:-1]}\n")

    print("Percentage of errors in multiple components")
    print(f"Device level:\t\t {str(fig_header)[1:-1]}")
    print(
        f"# of error levels:\t {str(np.array(level_data)[:, 0])[1:-1]}\n# of multi-components:\t {str(np.array(level_data)[:, 1])[1:-1]}")
    print(f"Percentage(%):\t\t {str(data[1])[1:-1]}\n")


    num_columns = data.shape[1]
    column_indices = np.arange(num_columns)
    width = 0.4  # 设置柱的宽度
    loction = [-1 * width / 2 - 0.02, width / 2 + 0.02]
    for i in range(2):
        # plt.bar(column_indices + loction[i], np.ones(num_columns), width, color=use_color[i*2],
        #        hatch=bar_patters[i],edgecolor="black")
        bars = plt.bar(column_indices + loction[i], data[i, :], width, label=weight_name[i], color=use_color[i],
                       edgecolor="black", hatch=bar_patters[i])
        for bar in bars:
            yval = bar.get_height()
            if 99 < yval < 99.1:
                plt.text(bar.get_x() + bar.get_width() / 2, yval, '99.0', ha='center', va='bottom',
                         fontsize=25)
            elif yval == 50:
                plt.text(bar.get_x() + bar.get_width() / 2, yval, '50.0', ha='center', va='bottom',
                         fontsize=25)
            else:
                plt.text(bar.get_x() + bar.get_width() / 2, yval, '{:.3g}'.format(yval), ha='center', va='bottom',
                         fontsize=26)

    plt.xticks(fontsize=40)
    plt.yticks(fontsize=33)
    plt.ylim(0, 108)
    plt.ylabel(y_label, fontsize=40)
    plt.xticks(column_indices, header)
    plt.legend(loc="upper center", fontsize=40, bbox_to_anchor=(0.5, 1.17), ncol=2, frameon=False, labelspacing=0.3,
               handletextpad=0.2, handlelength=2, columnspacing=0.3)

    if save_figures:
        fig = plt.gcf()
        fig.set_size_inches(23, 8)
        plt.savefig(save_figures, bbox_inches='tight')
    print(f"Figure.2 is saved in {save_figures.absolute()}")
    #plt.show()


def same_level_proportion(fault_data, is_drop=True):
    # only the spatial information is reserved
    fault_data = fault_data[spatial_header]
    fault_data = fault_data.drop_duplicates()
    res_level = []
    res_cell_error = []
    for i in range(len(spatial_header)):
        if i == len(spatial_header) - 1:
            # row
            tmp_header = spatial_header[:i - 1] + spatial_header[i:i + 1]
            sub_header = spatial_header[-2]
        else:
            # 其他
            tmp_header = spatial_header[:i + 1]
            sub_header = spatial_header[i + 1]

        if i == len(spatial_header) - 3:
            # bank的子结构算的是cell
            tmp_data = fault_data[spatial_header[:i + 3]]
            if is_drop:
                drop_tmp_data = tmp_data.drop_duplicates()
                drop_tmp_data = drop_tmp_data[spatial_header[:i + 2]]
            else:
                drop_tmp_data = tmp_data[spatial_header[:i + 2]]
        else:
            tmp_data = fault_data[spatial_header[:i + 2]]
            if is_drop:
                drop_tmp_data = tmp_data.drop_duplicates()
            else:
                drop_tmp_data = tmp_data

        group_tmp_data = drop_tmp_data.groupby(tmp_header, as_index=False).count()

        all_level_num = len(group_tmp_data)
        multi_group_tmp_data = group_tmp_data[group_tmp_data[sub_header] > 1]
        level_with_multi_num = len(multi_group_tmp_data)
        if not is_drop:

            all_level_cells_num = group_tmp_data[sub_header].sum()
            level_with_multi_cells_num = multi_group_tmp_data[sub_header].sum()
            res_cell_error.append([all_level_cells_num, level_with_multi_cells_num])
            res_level.append([all_level_num, level_with_multi_num])
        else:
            res_level.append([all_level_num, level_with_multi_num])
    if not is_drop:
        return res_level, res_cell_error
    else:
        return res_level


if __name__ == "__main__":
    save_fold = Path(r"D:\code\HBMErrors\final_used")
    save_file = save_fold.joinpath(r"spatial_fault_data1.csv")
    result_file_fold = save_fold.joinpath(r"result")
    result_file_fold.mkdir(exist_ok=True)

    fault_data = pd.read_csv(save_file)

    # get # of device levels with multiple component error
    res_level_drop = same_level_proportion(fault_data, is_drop=True)
    # get  # of cells in the device levels with multiple errors
    res_level, res_cell_error = same_level_proportion(fault_data, is_drop=False)

    # fig information
    fig_header = ["DSA", "HBM", "SID", "PS-CH", "BG", "Bank", "Col", "Row"]
    drop_name = ["Individually occur.", "Occur with other sublevels."]
    res_name = ["Only one cell has errors.", "Multiple cells have errors."]
    res_cell_name = ["Individually occur.", "Occur with other cells."]
    fig_title_cell = "(a)"  # "(a)Percentage of cells with errors\n across various levels."
    fig_title_level = "(b)"  # "(b)Percentage of sublevels with errors\n across various levels."
    bar_name = ["Errors in multiple cells", "Errors in multiple components"]
    # output results and plot fig
    print("The results of Figure 2 in Sec-3.2")
    plot_figures_merge(res_level_drop[2:], res_cell_error[2:], bar_name, fig_header,
                       use_color=['#95B68E', '#E5E5E5', '#B6C7EA', '#E5E5E5'],
                       save_figures=result_file_fold.joinpath("merge_spatial.pdf"))