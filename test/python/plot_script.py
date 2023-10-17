import matplotlib.pyplot as plt
import numpy as np
import os

import samples
from profile_script import (profile_type_list, shape_list,
                            written_data_directory, written_data_label)

plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.marker'] = 'o'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams["legend.frameon"] = False
plt.rcParams['figure.titlesize'] = 'x-large'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams["figure.figsize"] = (8, 6)


def line_chart(shape_list, profile_type_list, plot_target,
               plot_type_list, ylabel, file_label, plot_logy=False):
    typeid_list = [find_type_index(profile_type_list, type) for type in plot_type_list]
    assert all(idx != -1 for idx in typeid_list)
    x_tick_num = len(shape_list)
    x_tick_position = np.linspace(min(shape_list), max(shape_list), x_tick_num)

    def line_chart_plot_axis(ax, ydata_dict, plot_type, ylabel, use_logy=False):
        for data_label, ydata in ydata_dict.items():
            for i, instance in enumerate(plot_type):
                label = instance.__name__ + "_" + data_label
                if use_logy:
                    ax.semilogy(x_tick_position, ydata[typeid_list[i]], label=label)
                else:
                    ax.plot(x_tick_position, ydata[typeid_list[i]], label=label)
        ax.set_xticks(x_tick_position)
        ax.set_xticklabels(shape_list)
        ax.legend(bbox_to_anchor=(0.99, 1.03), loc='upper left')
        ax.set_xlabel("Matrix Dimension(M=N=K)")
        ax.set_ylabel(ylabel)
        return ax

    fig = plt.figure()
    if (plot_logy == False):
        ax = fig.subplots(1, 1)
        ax = line_chart_plot_axis(ax, plot_target,
                                  plot_type_list, ylabel, use_logy=False)
    else:
        axes = fig.subplots(2, 1, sharex=True)
        axes[0] = line_chart_plot_axis(axes[0], plot_target,
                                       plot_type_list, ylabel, use_logy=False)
        axes[1] = line_chart_plot_axis(axes[1], plot_target,
                                       plot_type_list, "Log " + ylabel, use_logy=True)
        axes[1].get_legend().remove()
    fig.autofmt_xdate()
    fig.suptitle("Kernel Performance Comparison on RTX3090", y=0.92)
    fig.savefig(image_directory_path + f"line_chart_{file_label}.png", bbox_inches='tight')


def pie_chart(shape_list, profile_type_list, plot_target,
              plot_type, subplot_row=2, subplot_col=4):
    typeidx = find_type_index(profile_type_list, plot_type)
    assert typeidx != -1
    colors = ('#ff9999', '#66b3ff', '#99ff99')
    explode = (0.3, 0.0, 0.1)

    def pie_chart_axis(ax, proportion, title):
        ax.pie(proportion, explode=explode, autopct='%1.1f%%',
               colors=colors, shadow=True, startangle=90)
        ax.set_title(title, y=-0.1)
        ax.axis('equal')
        return ax

    fig_num = int(np.ceil(len(shape_list) / (subplot_row * subplot_col)))
    for n in range(fig_num):
        fig = plt.figure()
        axes = fig.subplots(subplot_row, subplot_col)
        offset = n * subplot_row * subplot_col
        for i, ax in enumerate(axes.flat):
            if (offset + i) < len(shape_list):
                shapeidx = offset + i
                proportion = plot_target(typeidx, shapeidx).values()
                title = "shape: " + str(shape_list[offset + i])
                ax = pie_chart_axis(ax, proportion, title)
            else:
                ax.remove()

        fig.legend(plot_target(typeidx, shapeidx).keys())
        fig.suptitle(plot_type.__name__, y=0.92)
        fig.autofmt_xdate()
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(image_directory_path + f"pie_chart_{plot_type.__name__}_{n}.png")


def reader(data_dict, data_label, written_data_directory):
    for label in data_label:
        with open(written_data_directory + label + ".txt", "r") as f:
            data_dict[label] = np.loadtxt(f)


def find_type_index(type_list, type):
    return next(
        (i for i, type_obj in enumerate(type_list) if type_obj == type),
        -1
    )


if __name__ == "__main__":

    # written_data_directory = "test/data/"
    image_directory_path = "test/image/"
    if not os.path.exists(image_directory_path):
        os.makedirs(image_directory_path)

    fetched_data_dict = {}

    reader(fetched_data_dict, written_data_label, written_data_directory)

    # ---------------------------

    line_chart_target_label = [
        "total",
        "matmul",
        "quantization",
        "dequantization",
        # "performance",
    ]

    line_chart_type_list = [
        samples.Int4MatmulInt32Out,
        samples.Int4SpMatmulInt32Out,
        samples.Int8MatmulInt32Out,
        samples.Int8SpMatmulInt32Out,
        # samples.FP16Matmul,
        # samples.FP32Matmul,
        samples.Int8SpmmCuspLtFp16Out,
        # samples.Int8SpMatmulInt8Out,
        # samples.Int8MatmulInt8Out,
    ]

    ylabel = "Time"
    plot_logy = True

    assert (all(item in written_data_label for item in line_chart_target_label))

    line_chart_target = {label: fetched_data_dict[label] for label in line_chart_target_label}

    assert (all(item in profile_type_list for item in line_chart_type_list))

    for type in line_chart_type_list:
        file_label = type.__name__
        line_chart(shape_list, profile_type_list, line_chart_target,
                   [type], ylabel, file_label, plot_logy)

    # ---------------------------

    line_chart_target_label = [
        # "total", 
        # "matmul", 
        # "quantization", 
        # "dequantization",
        "performance",
    ]

    line_chart_type_list = [
        # samples.Int4MatmulInt32Out,
        # samples.Int4SpMatmulInt32Out,
        samples.Int8SpmmCuspLtFp16Out,
        samples.Int8SpMatmulInt32Out,
        samples.Int8MatmulInt32Out,
        # samples.FP16Matmul,
        # samples.FP32Matmul,
    ]

    file_label = "performance_int8"
    ylabel = "TFLOPS"
    plot_logy = False

    assert (all(item in written_data_label for item in line_chart_target_label))

    line_chart_target = {label: fetched_data_dict[label] for label in line_chart_target_label}

    assert (all(item in profile_type_list for item in line_chart_type_list))

    line_chart(shape_list, profile_type_list, line_chart_target,
               line_chart_type_list, ylabel, file_label, plot_logy)
    # ---------------------------

    line_chart_target_label = [
        # "total", 
        # "matmul", 
        # "quantization", 
        # "dequantization",
        "performance",
    ]

    line_chart_type_list = [
        samples.Int4SpMatmulInt32Out,
        samples.Int4MatmulInt32Out,
        samples.Int8SpmmCuspLtFp16Out,
        samples.Int8SpMatmulInt32Out,
        samples.Int8MatmulInt32Out,
        samples.FP16Matmul,
        samples.FP32Matmul,
        # samples.Int8SpMatmulInt8Out,
        # samples.Int8MatmulInt8Out,
    ]

    file_label = "performance"
    ylabel = "TFLOPS"
    plot_logy = False

    assert (all(item in written_data_label for item in line_chart_target_label))

    line_chart_target = {label: fetched_data_dict[label] for label in line_chart_target_label}

    assert (all(item in profile_type_list for item in line_chart_type_list))

    line_chart(shape_list, profile_type_list, line_chart_target,
               line_chart_type_list, ylabel, file_label, plot_logy)

    # ---------------------------

    pie_chart_target_label = {
        "quantization",
        "matmul",
        "dequantization",
    }

    pie_chart_type_list = [
        samples.Int4SpMatmulInt32Out,
        samples.Int4MatmulInt32Out,
        samples.Int8SpmmCuspLtFp16Out,
        samples.Int8SpMatmulInt32Out,
        samples.Int8MatmulInt32Out,
    ]

    subplot_row = 2
    subplot_col = 4

    assert (all(item in written_data_label for item in pie_chart_target_label))

    pie_chart_target = lambda typeidx, shapeidx: {
        label: fetched_data_dict[label][typeidx][shapeidx] for label in pie_chart_target_label
    }

    assert (all(item in profile_type_list for item in pie_chart_type_list))

    for pie_chart_type in pie_chart_type_list:
        pie_chart(shape_list, profile_type_list, pie_chart_target,
                  pie_chart_type, subplot_row, subplot_col)
