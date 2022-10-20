from os import error, listdir, name, terminal_size
from os.path import isfile, join
from numpy import average, mean, std

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker

def plot_clustered_bars(title, x_datas, y_datas, y_labels):
   
    width = 1/len(y_datas)  # the width of the bars
    from itertools import chain
    ticks = list(np.unique(np.concatenate(x_datas)))
    fig, ax = plt.subplots(figsize=(10,4))

    for i, (x, y, y_label) in enumerate(zip(x_datas, y_datas, y_labels)):
        ax.bar(x + (i - len(y_datas)/2)*width, y, width, label=y_label)

    plt.grid(axis='y')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Latency [us]')
    #ax.set_title(title)
    ax.set_xlabel('Message Size [KB]')
    ax.set_xticks(ticks)
    ax.legend()
    plt.show()
    #plt.savefig(f"{title}.png")


def plot_lines(title, x_datas, y_datas, y_series_labels, y_styles=None, logx=True, logy=True, x_label='Message Size', y_label='Latency [us]', y_errors=None, legend_loc=None, throughput=False):
    if not(y_styles):
        y_styles = [None for _ in range(len(y_series_labels))]
    
    if not(y_errors):
        y_errors = [None for _ in range(len(y_series_labels))]

    fig, ax = plt.subplots(figsize=(9,6))
    series  = []
    for x, y, y_series_label, y_style, y_error in zip(x_datas, y_datas, y_series_labels, y_styles, y_errors):
        if y_style:
            if not y_error is None:
                series.append(ax.errorbar(x, y,  yerr = y_error, fmt=y_style, label=y_series_label, capsize=4.0, linewidth=3, markersize=8, markeredgewidth=3))
            else:
                line, = ax.plot(x, y, y_style, label=y_series_label, linewidth=3, markersize=8, markeredgewidth=3)
                series.append(line)
        else:
            if not y_error is None:
                series.append(ax.errorbar(x, y,  yerr = y_error, fmt=y_style, label=y_series_label, capsize=4.0, linewidth=3, markersize=8, markeredgewidth=3))
            else:
                line, = ax.plot(x, y, label=y_series_label, linewidth=3, markersize=8, markeredgewidth=3)
                series.append(line)

    plt.grid(axis='y')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    if throughput:
        ax.set_ylabel('Throughput [Gbps]', fontsize=20)
        ax.axis(ymin=0,ymax=100)
    else:
        ax.set_ylabel(y_label,  fontsize=20)
        # ax.axis(ymin=1)
    #ax.set_title(title)
    if logy:
        ax.set_yscale('log')
    
    if logx:
        ax.set_xscale('log', base=2)
        
    if legend_loc is None :
        if logy:
            ax.legend(series, y_series_labels, loc="lower right", handlelength=4)
        else:
            ax.legend(series, y_series_labels, loc="upper left", handlelength=4)
    else:
        ax.legend(    series, y_series_labels, loc=legend_loc, fontsize=14, handlelength=4)

    if x_label == "Message Size":
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: sizeof_fmt(y)))
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(fontsize=18)
    ax.set_xlabel(x_label, fontsize=20)
    # plt.show()
    plt.savefig(f"{title}.png", format='png', bbox_inches='tight')

def plot_lines2(title, x_datas, y_datas, y_labels, y_styles=None, logx=True, logy=True, y_errors=None):
    if not(y_styles):
        y_styles = [None for _ in range(len(y_labels))]
    
    if not(y_errors):
        y_errors = [None for _ in range(len(y_labels))]

    fig, ax = plt.subplots(figsize=(7,6))

    for x, y, y_label, y_style, y_error in zip(x_datas, y_datas, y_labels, y_styles, y_errors):
        if y_style:
            ax.plot(x, y, y_style, label=y_label)
        else:
            ax.plot(x, y, label=y_label)
        
        if y_error is not None:
            ax.fill_between(x,y-y_error,y+y_error,alpha=.1)

    plt.grid(axis='y')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Latency [us]')
    #ax.set_title(title)
    if logy:
        ax.set_yscale('log')
        ax.legend(loc="lower right")
    else:
        ax.legend(loc="upper left")
        
    if logx:
        ax.set_xscale('log', base=2)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: sizeof_fmt(y)))
    plt.xticks(rotation=0)
    ax.set_xlabel('Message Size')
    plt.show()
    plt.savefig(f"{title}.png", format='png')

def plot_lines3(title, x_datas, y_datas, y_labels, y_styles=None, logx=True, logy=True, y_errors=None):
    if not(y_styles):
        y_styles = [None for _ in range(len(y_labels))]
    
    if not(y_errors):
        y_errors = [None for _ in range(len(y_labels))]

    fig, ax = plt.subplots(figsize=(7,6))

    for x, y, y_label, y_style, y_error in zip(x_datas, y_datas, y_labels, y_styles, y_errors):
        if y_style:
            if not y_error is None:
                ax.errorbar(x, y,  yerr = y_error, fmt=y_style, label=y_label, capsize=2.0, linewidth=1)
            else:
                ax.plot(x, y, y_style, label=y_label)
        else:
            if not y_error is None:
                ax.errorbar(x, y,  yerr = y_error, fmt=y_style, label=y_label, capsize=2.0, linewidth=1)
            else:
                ax.plot(x, y, label=y_label)
        
        if y_error is not None:
            ax.fill_between(x,y-y_error,y+y_error,alpha=.1)

    plt.grid(axis='y')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Latency [us]')
    #ax.set_title(title)
    if logy:
        ax.set_yscale('log')
        ax.legend(loc="lower right")
    else:
        ax.legend(loc="upper left")
        
    if logx:
        ax.set_xscale('log', base=2)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: sizeof_fmt(y)))
    plt.xticks(rotation=0)
    ax.set_xlabel('Message Size')
    plt.show()
    plt.savefig(f"{title}.png", format='png')

def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return "%3.f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


# def normality_test(df_accl):
#     df_accl              = df_accl[ df_accl["rank_id"] == 0]
#     collectives     = df_accl["collective"].unique()
#     for collective in collectives:
#         subset              = df_accl[(df_accl["collective"] == collective) ]
#         buffer_sizes        = subset["size[B]"].unique()
#         segmentation_sizes  = subset["segment_size[B]"].unique() 
#         grouped             = subset.groupby(["board_instance", "number_of_banks", "size[B]", "segment_size[B]"])
        
#         import scipy.stats as stats
#         import math
#         for (board_instance, banks, bsize, ssize), group in grouped:
#             print(bsize, ssize, group)
#             exe     = group['Latency [us]'].to_numpy()
#             #bufsize = grouped["size[B]"].to_numpy()
#             plt.clf()
#             plt.hist(exe, bins=20)
#             mu      = mean(exe)
#             sigma   = std(exe, ddof=1)
#             x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
#             plt.twinx()
#             plt.plot(x, stats.norm.pdf_accl(x, mu, sigma), "r--")
#             # normality test
#             stat, p = stats.shapiro(exe)
#             print('Statistics=%.3f, p=%.3f' % (stat, p))
#             # interpret
#             alpha = 0.05
#             if p > alpha:
#                 print('Sample looks Gaussian (fail to reject H0)')
#             else:
#                 print('Sample does not look Gaussian (reject H0)')
#             plt.title(f"{board_instance} #banks{banks} {collective} b{bsize} s{ssize}\n"+("sample "+  ("looks" if p > alpha else "doesn't look" ) +" Gaussian"))
#             plt.show()


# def compare_ssize(df_accl):
#     df_accl              = df_accl[ df_accl["rank_id"] == 0]
#     collectives     = df_accl["collective"].unique()
#     for collective in collectives:
#         subset              = df_accl[(df_accl["collective"] == collective) & (df_accl["experiment"] == "test1") & (df_accl["board_instance"]=="xilinx_u280_xdma_201920_3") & (df_accl["number_of_banks"] == 6)]
#         grouped             = subset.groupby(["segment_size[B]", "size[B]"]).mean(['execution_time[us]', 'execution_time_fullpath[us]'])
#         grouped.reset_index(inplace=True)
#         print("grouped", grouped)
#         grouped             = grouped.groupby(["segment_size[B]"])

#         series_label = []
#         series_y     = []
#         series_x     = []
#         for ssize, group in grouped:
#             exe     = group['execution_time[us]'].to_numpy()
#             bufsize = group['size[B]'].to_numpy()*1024
#             #print("group", ssize,  group)
#             #print(exe)
#             #print(bufsize)
#             series_label.append(ssize)
#             series_y.append(exe)
#             series_x.append(bufsize)
#         plot_clustered_bars(collective, series_x, series_y, series_label)

# def get_statistics(df_accl):
#     df_accl              = df_accl[ df_accl["rank_id"] == 0]
#     collectives     = df_accl["collective"].unique()

#     seg_sizes = (2048,1024,512,256,128)
#     data_sizes = (1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768)
#     # check sendrecv u280
#     subset = df_accl[(df_accl["experiment"] == "u280_sendrecv") & (df_accl["collective"] == "Send/recv") & (df_accl["board_instance"] == "xilinx_u280_xdma_201920_3")]
#     # print(subset)
#     # for seg_size in seg_sizes:
#     #     for data_size in data_sizes:
#     #         subsubset = subset[(subset["segment_size[B]"] == seg_size) & (subset["number_of_banks"] == 6) & (subset["size[B]"] == data_size)]
#     #         # print(subsubset)
#     #         mean = subsubset["throughput[Gbps]"].mean()
#     #         count = subsubset["throughput[Gbps]"].count()
#     #         std = subsubset["throughput[Gbps]"].std()
#     #         if count < 100:
#     #             print(f"U280, sendrecv, seg size:{seg_size}, num banks:{6}, data size:{data_size}, mean:{mean}, std:{std}, count:{count}")

#     # for num_bank in range (2,6):
#     #     for data_size in data_sizes:
#     #         subsubset = subset[(subset["segment_size[B]"] == 1024) & (subset["number_of_banks"] == num_bank) & (subset["size[B]"] == data_size) ]
#     #         mean = subsubset["throughput[Gbps]"].mean()
#     #         count = subsubset["throughput[Gbps]"].count()
#     #         std = subsubset["throughput[Gbps]"].std()
#     #         if count < 100:
#     #             print(f"U280, sendrecv, seg size:{1024}, num banks:{num_bank}, data size:{data_size}, mean:{mean}, std:{std}, count:{count}")

    
#     # for seg_size in seg_sizes:
#     #     for num_bank in range (2,7):
#     #         for data_size in data_sizes:
#     #             subsubset = subset[(subset["segment_size[B]"] == seg_size) & (subset["number_of_banks"] == num_bank) & (subset["size[B]"] == data_size)]
#     #             # print(subsubset)
#     #             mean = subsubset["throughput[Gbps]"].mean()
#     #             count = subsubset["throughput[Gbps]"].count()
#     #             std = subsubset["throughput[Gbps]"].std()
#     #             # if count < 100:
#     #             print(f"U280, sendrecv, seg size:{seg_size}, num banks:{num_bank}, data size:{data_size}, mean:{mean}, std:{std}, count:{count}")

#     num_nodes = (3,4)
#     #check U280 collectives (<= 4 nodes)
#     subset = df_accl[ (df_accl["board_instance"] == "xilinx_u280_xdma_201920_3") & (df_accl["collective"] != "Send/recv") & (df_accl["number_of_nodes"] <= 4 )]
#     for num_node in num_nodes:
#         for collective in collectives:
#             if collective == "Allreduce":
#             # if collective != "Send/recv":
#                 for data_size in data_sizes:
#                     subsubset = subset[(subset["collective"] == collective) &(subset["segment_size[B]"] == 1024) & (subset["number_of_banks"] == 6) & (subset["size[B]"] == data_size) & (subset["number_of_nodes"] == num_node)]
#                     # print(subsubset)
#                     mean = subsubset["execution_time[us]"].mean()
#                     count = subsubset["execution_time[us]"].count()
#                     std = subsubset["execution_time[us]"].std()
#                     if count < 100:
#                         print(f"U280, {collective}, num nodes:{num_node}, seg size:{1024}, num banks:{6}, data size:{data_size}, mean:{mean}, std:{std}, count:{count}")

#     #check U250 collectives (<= 4 nodes)
#     # subset = df_accl[(df_accl["board_instance"] == "xilinx_u250_gen3x16_xdma_shell_3_1") & (df_accl["collective"] != "Send/recv") & (df_accl["number_of_nodes"] <= 4 )]
#     # # print(subset)
#     # for num_node in num_nodes:
#     #     for collective in collectives:
#     #         if collective != "Send/recv":
#     #             for data_size in data_sizes:
#     #                 subsubset = subset[(subset["collective"] == collective) &(subset["segment_size[B]"] == 1024) & (subset["number_of_banks"] == 3) & (subset["size[B]"] == data_size) & (subset["number_of_nodes"] == num_node)]
#     #                 # print(subsubset)
#     #                 mean = subsubset["execution_time[us]"].mean()
#     #                 count = subsubset["execution_time[us]"].count()
#     #                 std = subsubset["execution_time[us]"].std()
#     #                 print(f"U250, {collective}, num nodes:{num_node}, seg size:{1024}, num banks:{3}, data size:{data_size}, mean:{mean}, std:{std}, count:{count}")
        
#     # check collectives (> 4 nodes)
#     num_nodes = (5,6,7,8)
#     subset = df_accl[(df_accl["collective"] != "Send/recv") & (df_accl["number_of_nodes"] > 4 )]
#     for num_node in num_nodes:
#         for collective in collectives:
#             if collective == "Allreduce":
#             # if collective != "Send/recv":
#                 for data_size in data_sizes:
#                     subsubset = subset[(subset["collective"] == collective) & (subset["segment_size[B]"] == 1024) & (subset["size[B]"] == data_size) & (subset["number_of_nodes"] == num_node)]
#                     # print(subsubset)
#                     mean = subsubset["execution_time[us]"].mean()
#                     count = subsubset["execution_time[us]"].count()
#                     std = subsubset["execution_time[us]"].std()
#                     print(f"{collective}, num nodes:{num_node}, seg size:{1024}, data size:{data_size}, mean:{mean}, std:{std}, count:{count}")

# def compare_openMPI(df_accl, H2H=True, F2F=True, error=False):
#     df_accl              = df_accl[ (df_accl["rank_id"] == 0) & ( (df_accl["number_of_nodes"]==4)  | (df_accl["collective"] == "Send/recv"))]
#     collectives     = df_accl["collective"].unique()
#     segment_size    = 1024
#     for collective in collectives:
#         print(collective)
#         subset              = df_accl[(df_accl["collective"] == collective) & (df_accl["segment_size[B]"] == segment_size) &
#                                     (  
#                                     ( ( df_accl["board_instance"] == "xilinx_u280_xdma_201920_3") & (df_accl["number_of_banks"] == 6))                                   
#                                     )]
#         grouped             = subset.groupby(["board_instance", "size[B]"]).agg({'execution_time[us]':['mean','std'], 'execution_time_fullpath[us]':['mean','std']})
#         grouped.reset_index(inplace=True)
#         grouped             = grouped.groupby(["board_instance"])
#         series_label = []
#         series_y     = []
#         series_x     = []
#         styles       = []
#         stdevs       = []
#         average_delta = None
#         i = 0
#         for board, group in grouped:
#             print(group)
#             exe          = group['execution_time[us]']['mean'].to_numpy()
#             exe_std      = group['execution_time[us]']['std'].to_numpy()
#             bufsize      = group['size[B]'].to_numpy()*1024
#             exe_full     = group['execution_time_fullpath[us]']['mean'].to_numpy()
#             exe_full_std = group['execution_time_fullpath[us]']['std'].to_numpy()

#             board = simplify_board_name(board)
#             i+=1
#             if np.any(exe != 0) and F2F:
#                 series_label.append(f"ACCL F2F")
#                 series_y.append(exe)
#                 series_x.append(bufsize)
#                 stdevs.append(exe_std)
#                 styles.append(f"C{2}+-")
#             if np.any(exe_full != 0) and H2H:
#                 series_label.append(f"ACCL H2H")
#                 series_y.append(exe_full)
#                 series_x.append(bufsize)
#                 stdevs.append(exe_full_std)
#                 styles.append(f"C{2}+--")

#             if board.find("U280") != -1:
#                 average_delta = np.abs(exe_full - exe)
#         #For OpenMPI
#         subset              = df_accl[(df_accl["collective"] == collective) & (df_accl["board_instance"].str.contains("OpenMPI") )]
#         grouped             = subset.groupby(["board_instance", "size[B]"]).agg({'execution_time[us]':['mean','std'], 'execution_time_fullpath[us]':['mean','std']})
#         grouped.reset_index(inplace=True)
#         grouped             = grouped.groupby(["board_instance"])
#         i = 2
#         for board, group in grouped:
#             exe          = group['execution_time[us]']['mean'].to_numpy()
#             exe_std      = group['execution_time[us]']['std'].to_numpy()
#             bufsize      = group['size[B]'].to_numpy()*1024
#             exe_full     = group['execution_time_fullpath[us]']['mean'].to_numpy()
#             exe_full_std = group['execution_time_fullpath[us]']['std'].to_numpy()

#             board = simplify_board_name(board)
#             i+=1
            
#             if np.any(exe) and F2F:
#                 series_label.append(f"{board}")
#                 series_y.append(exe)
#                 series_x.append(bufsize)
#                 stdevs.append(None)
#                 styles.append(f"C{i}+-")
#             if np.any(exe_full != 0) and H2H:
#                 series_label.append(f"{board} H2H")
#                 series_y.append(exe_full)
#                 series_x.append(bufsize)
#                 stdevs.append(exe_full_std)
#                 styles.append(f"C{i}+--")

        
#         plot_lines("compare_OMPI"+("H2H" if H2H else "") + ("F2F" if F2F else "")+collective.replace("/", ""), series_x, series_y, series_label, styles, y_label='Latency [us]', logx=True, legend_loc ="upper left", y_errors=(stdevs if error else None))


def compare_latency(df_accl, df_mpi, number_of_nodes=4, error=False):
    df_accl              = df_accl[ (df_accl["rank_id"] == 0) & (df_accl["number_of_nodes"]==number_of_nodes)]
    collectives     = df_accl['collective'].apply(lambda r: '_'.join(r.split('_')[:-1])).unique()
    print(collectives)
    segment_size    = 4*1024*1024
    for collective in collectives:
        subset              = df_accl[(df_accl["collective"].str.startswith(collective)) & (df_accl["segment_size[B]"] == segment_size) & (df_accl["number_of_banks"] == 6) ]
        # print(subset)
        grouped             = subset.groupby(["collective", "size[B]"]).agg({'execution_time[us]':['mean','std']})
        grouped.reset_index(inplace=True)
        grouped             = grouped.groupby(["collective"])
        series_label = []
        series_y     = []
        series_x     = []
        styles       = []
        stdevs       = []
        i = 0
        for coll, group in grouped:
            # print(group)
            exe          = group['execution_time[us]']['mean'].to_numpy()
            exe_std      = group['execution_time[us]']['std'].to_numpy()
            bufsize      = group['size[B]'].to_numpy()
            
            if np.any(exe != 0):
                series_label.append(f"ACCL {coll}")
                series_y.append(exe)
                series_x.append(bufsize)
                stdevs.append(exe_std)
                styles.append(f"C{i+1}+-")
                i+=1


        #For OpenMPI
        subset              = df_mpi[(df_mpi["collective name"].str.startswith(collective)) & (df_mpi["board_instance"]=="OpenMPI")]
        grouped             = subset.groupby(["board_instance", "buffer size[KB]"]).agg({'execution_time[us]':['mean','std'], 'execution_time_fullpath[us]':['mean','std']})
        grouped.reset_index(inplace=True)
        grouped             = grouped.groupby(["board_instance"])
        for board, group in grouped:
            exe          = group['execution_time[us]']['mean'].to_numpy()
            exe_std      = group['execution_time[us]']['std'].to_numpy()
            bufsize      = group['buffer size[KB]'].to_numpy()*1024
            exe_full     = group['execution_time_fullpath[us]']['mean'].to_numpy()
            exe_full_std = group['execution_time_fullpath[us]']['std'].to_numpy()

            # if np.any(exe):
            #     series_label.append(f"{board}")
            #     series_y.append(exe)
            #     series_x.append(bufsize)
            #     stdevs.append(None)
            #     styles.append(f"C{i}+-")
            if np.any(exe_full != 0):
                series_label.append(f"{board}_H2H")
                series_y.append(exe_full)
                series_x.append(bufsize)
                stdevs.append(exe_full_std)
                styles.append(f"C{i+1}+-")
                i+=1

        
        plot_lines("compare_latency_"+collective.replace("/", "")+"_nr_"+str(number_of_nodes), series_x, series_y, series_label, styles, y_label='Latency [us]', logx=True, legend_loc ="upper left", y_errors=(stdevs if error else None))


# def plot_box_plot(title,xs, ys, y_labels, xlabel, ylabel, logy=False, logx=True):

#     def set_box_color(bp, color):
#         plt.setp(bp['boxes'],    color=color)
#         plt.setp(bp['whiskers'], color=color)
#         plt.setp(bp['caps'],     color=color)
#         plt.setp(bp['medians'],  color=color)
#     colours = ['#d7191c', '#fdae61', '#abd9e9', '#2c7bb6', "#7b3294"]
#     plt.figure()
#     if xs is None:
#         xs = [None for _ in y_labels]
#     for (i, ( y,x,  y_label, colour)) in enumerate(zip( ys,xs, y_labels, colours)):
#         #positions = np.array(range(i, len(y)*len(ys), len(ys)))
#         #print(positions)
#         if x is not None:
#             bpl = plt.boxplot(y, positions=x, widths=1/len(ys))
#         else:
#             bpl = plt.boxplot(y,  widths=1/len(ys))
#         set_box_color(bpl, colour) 
#         plt.plot([], c=colour, label=y_label)

#     plt.ylabel(ylabel,  fontsize=20)
#     plt.xlabel(xlabel,  fontsize=20)
#     #ax.set_title(title)
#     if logy:
#         plt.yscale('log')
#     if logx:
#         plt.xscale('log', base=2)
#     if xlabel == "Message Size":
#         plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: sizeof_fmt(y)))
#     plt.legend()

#     #plt.xticks(range(0, len(xs[0]) * 2, 2), xs[0])
#     #plt.xlim(-2, len(xs[0])*2)
#     plt.savefig(f"{title}.png", format='png',bbox_inches='tight')

# def compare_box_plot(df_accl, F2F=True, H2H=True):
#     selection_params = [{"collective":"Send/recv",
#                         "segment_size[B]":1024,
#                         "board_instance":"xilinx_u280_xdma_201920_3", 
#                         "number_of_banks":6},
#                         {"collective":"Send/recv",
#                         "segment_size[B]":1024,
#                         "board_instance":"xilinx_u250_gen3x16_xdma_shell_3_1", 
#                         "number_of_banks":3}, 
#                         {"collective":"Send/recv",
#                         "segment_size[B]":0,
#                         "board_instance":"OpenMPI", 
#                         "number_of_banks":0}]
#     series_label    = []
#     series_y        = []
#     series_x        = []

#     for selection_param in selection_params:

#         collective_name = selection_param["collective"]
#         seg_size        = selection_param["segment_size[B]"]
#         board           = selection_param["board_instance"]
#         num_banks       = selection_param["number_of_banks"]

#         subset              = df_accl[   (df_accl["rank_id"]         == 0) & 
#                                     (df_accl["collective"] == collective_name) & (df_accl["segment_size[B]"] == seg_size ) &
#                                     (df_accl["board_instance"]  == board)           & (df_accl["number_of_banks"]  == num_banks)
#                                 ]
#         tmp_0y = []
#         tmp_0x = []
#         tmp_1y = []
#         tmp_1x = []
#         grouped             = subset.groupby(["size[B]"])
#         for bsize, group in grouped:
#             exe          = group['execution_time[us]'].to_numpy()
#             exe_full     = group['execution_time_fullpath[us]'].to_numpy()
#             board = simplify_board_name(board)
        
#             if np.any(exe != 0) and F2F:
#                 tmp_0y.append(exe)
#                 tmp_0x.append(bsize*1024)
#             if np.any(exe_full != 0) and H2H:
#                 tmp_1y.append(exe_full)
#                 tmp_1x.append(bsize*1024)
        
#         if F2F and len(tmp_0x) > 0:
#             series_label.append(f"{board} F2F")
#             series_y.append(tmp_0y)
#             series_x.append(tmp_0x)
#         if H2H  and len(tmp_1x) > 0:
#             series_label.append(f"{board} H2H")
#             series_y.append(tmp_1y)
#             series_x.append(tmp_1x)

#     plot_box_plot("send_recv_distribution", series_x, series_y, series_label,  "Message Size", "Latency [uS]", logy=True)

# def compare_box_plot_with_fixed_bsize(df_accl):
#     selection_params = [{"collective":"Send/recv",
#                         "segment_size[B]":1024,
#                         "board_instance":"xilinx_u280_xdma_201920_3", 
#                         "number_of_banks":6},
#                         {"collective":"Send/recv",
#                         "segment_size[B]":1024,
#                         "board_instance":"xilinx_u250_gen3x16_xdma_shell_3_1", 
#                         "number_of_banks":3}, 
#                         {"collective":"Send/recv",
#                         "segment_size[B]":0,
#                         "board_instance":"OpenMPI", 
#                         "number_of_banks":0}]

#     for bsize in df_accl["size[B]"].unique():
#         series_label    = []
#         series_y        = []
#         series_x        = []
#         i = 0
#         for selection_param in selection_params:

#             collective_name = selection_param["collective"]
#             seg_size        = selection_param["segment_size[B]"]
#             board           = selection_param["board_instance"]
#             num_banks       = selection_param["number_of_banks"]

#             subset              = df_accl[   (df_accl["rank_id"]         == 0)               & (df_accl["size[B]"]  == bsize)     &
#                                         (df_accl["collective"] == collective_name) & (df_accl["segment_size[B]"] == seg_size ) &
#                                         (df_accl["board_instance"]  == board)           & (df_accl["number_of_banks"]  == num_banks) ]
#             if board in ["xilinx_u280_xdma_201920_3","xilinx_u250_gen3x16_xdma_shell_3_1"]:
#                 F2F = True
#                 H2H = False
#             else:
#                 F2F = False
#                 H2H = True
#             print(subset)
#             tmp_0y = []
#             tmp_0x = []
#             tmp_1y = []
#             tmp_1x = []
#             grouped             = subset.groupby(["size[B]"])
#             for bsize, group in grouped:
#                 exe          = group['execution_time[us]'].to_numpy()
#                 exe_full     = group['execution_time_fullpath[us]'].to_numpy()
#                 print(board, F2F, H2H, exe)
#                 if np.any(exe != 0) and F2F:
#                     tmp_0y.append(exe)
#                     tmp_0x.append(i)
#                     i+=1
#                 if np.any(exe_full != 0) and H2H:
#                     tmp_1y.append(exe_full)
#                     tmp_1x.append(i)
#                     i+=1
            
#             board = simplify_board_name(board)
#             if len(tmp_0y) > 0:
#                 series_label.append(f"{board}")
#                 series_y.append(tmp_0y)
#                 series_x.append(tmp_0x)
#             if len(tmp_1y) > 0:
#                 series_label.append(f"{board}")
#                 series_y.append(tmp_1y)
#                 series_x.append(tmp_1x)

#         plot_box_plot(f"send_recv_distribution_bsize{bsize}", series_x, series_y, series_label,  "", "Latency [uS]", logy=False, logx=False)

# def compare_board(df_accl,H2H=True, F2F=True, error =False):
#     df_accl              = df_accl[ (df_accl["rank_id"] == 0) & ( (df_accl["number_of_nodes"]==4)  | (df_accl["collective"] == "Send/recv"))]
#     collectives     = df_accl["collective"].unique()
#     segment_size    = 1024
#     for collective in collectives:
#         print(collective)
#         subset              = df_accl[   (df_accl["collective"] == collective) & 
#                                     (
#                                         ((df_accl["segment_size[B]"] == segment_size) &
#                                             (  
#                                                 ( ( df_accl["board_instance"] == "xilinx_u280_xdma_201920_3") & (df_accl["number_of_banks"] == 6)) |
#                                                 ( ( df_accl["board_instance"] == "xilinx_u250_gen3x16_xdma_shell_3_1") & (df_accl["number_of_banks"] == 3))                                     
#                                             )
#                                         )| (df_accl["board_instance"] != "OpenMPI" ))]
#         grouped             = subset.groupby(["board_instance", "size[B]"]).agg({'execution_time[us]':['mean','std'], 'execution_time_fullpath[us]':['mean','std']})
#         grouped.reset_index(inplace=True)
#         grouped             = grouped.groupby(["board_instance"])
#         series_label = []
#         series_y     = []
#         series_x     = []
#         styles       = []
#         stdevs       = []

#         i = 0
#         for board, group in grouped:
#             print(group)
#             exe          = group['execution_time[us]']['mean'].to_numpy()
#             exe_std      = group['execution_time[us]']['std'].to_numpy()
#             bufsize      = group['size[B]'].to_numpy()*1024
#             exe_full     = group['execution_time_fullpath[us]']['mean'].to_numpy()
#             exe_full_std = group['execution_time_fullpath[us]']['std'].to_numpy()

#             board = simplify_board_name(board)
#             i+=1
#             if np.any(exe != 0) and F2F:
#                 series_label.append(f"ACCL {board} F2F")
#                 series_y.append(exe)
#                 series_x.append(bufsize)
#                 stdevs.append(exe_std)
#                 styles.append(f"C{i}-")
#             if np.any(exe_full != 0) and H2H:
#                 series_label.append(f"ACCL {board} H2H")
#                 series_y.append(exe_full)
#                 series_x.append(bufsize)
#                 stdevs.append(exe_full_std)
#                 styles.append(f"C{i}--")
        
#         plot_lines("board_comparison"+("H2H" if H2H else "") + ("F2F" if F2F else "")+collective.replace("/", ""), series_x, series_y, series_label, styles, y_label='Latency [us]', logx=True, legend_loc ="upper left", y_errors=(stdevs if error else None))
#         #plot_clustered_bars(collective, series_x, series_y, series_label)

# def sendrecv_banks(df_accl,H2H=False, F2F=True):
#     df_accl              = df_accl[(df_accl["experiment"] == "u280_sendrecv") & (df_accl["rank_id"] == 0) & (df_accl["number_of_nodes"]==2)  & (df_accl["collective"] == "Send/recv")]
#     collectives     = df_accl["collective"].unique()
#     segment_size    = 1024
#     for collective in collectives:
#         print(collective)
#         subset              = df_accl[(df_accl["collective"] == collective) & (df_accl["segment_size[B]"] == segment_size)]
#         grouped             = subset.groupby(["number_of_banks", "size[B]" ]).agg({'throughput[Gbps]':['mean','std'], 'throughput_fullpath[Gbps]':['mean','std']})
#         grouped.reset_index(inplace=True)
#         grouped             = grouped.groupby(["number_of_banks"])
#         series_label = []
#         series_y     = []
#         series_x     = []
#         styles       = []
#         stdevs       = []

#         i = 0
#         for banks, group in grouped:
#             print(group)
#             exe          = group['throughput[Gbps]']['mean'].to_numpy()
#             exe_std      = group['throughput[Gbps]']['std'].to_numpy()
#             bufsize      = group['size[B]'].to_numpy()*1024
#             exe_full     = group['throughput_fullpath[Gbps]']['mean'].to_numpy()
#             exe_full_std = group['throughput_fullpath[Gbps]']['std'].to_numpy()

#             i+=1
#             spare_buffer_bank = banks - 1
#             if np.any(exe != 0) and F2F:
#                 series_label.append(f"U280 {spare_buffer_bank} banks F2F")
#                 series_y.append(exe)
#                 series_x.append(bufsize)
#                 stdevs.append(exe_std)
#                 styles.append(f"C{i}-")
#             if np.any(exe_full != 0) and H2H:
#                 series_label.append(f"U280 {spare_buffer_bank} banks H2H")
#                 series_y.append(exe_full)
#                 series_x.append(bufsize)
#                 stdevs.append(exe_full_std)
#                 styles.append(f"C{i}--")

        
#         plot_lines("U280_Bank"+("H2H" if H2H else "") + ("F2F" if F2F else "")+collective.replace("/", ""), series_x, series_y, series_label, styles, y_errors=stdevs, logx=True, logy=False, throughput=True)
#         #plot_clustered_bars(collective, series_x, series_y, series_label)
            

# def sendrecv_segmentation(df_accl,H2H=False, F2F=True):
#     df_accl              = df_accl[(df_accl["experiment"] == "u280_sendrecv") & (df_accl["rank_id"] == 0) & (df_accl["number_of_nodes"]==2)  & (df_accl["collective"] == "Send/recv")]
#     collectives     = df_accl["collective"].unique()
#     segment_size    = (1024,512,256,128)
#     num_bank        = 6
#     for collective in collectives:
#         print(collective)
#         subset              = df_accl[(df_accl["collective"] == collective) & (df_accl["number_of_banks"] == num_bank)]
#         grouped             = subset.groupby(["segment_size[B]", "size[B]" ]).agg({'throughput[Gbps]':['mean','std'], 'throughput_fullpath[Gbps]':['mean','std']})
#         grouped.reset_index(inplace=True)
#         grouped             = grouped.groupby(["segment_size[B]"])
#         series_label = []
#         series_y     = []
#         series_x     = []
#         styles       = []
#         stdevs       = []

#         i = 0
#         for seg_size, group in grouped:
#             print(group)
#             exe          = group['throughput[Gbps]']['mean'].to_numpy()
#             exe_std      = group['throughput[Gbps]']['std'].to_numpy()
#             bufsize      = group['size[B]'].to_numpy()*1024
#             exe_full     = group['throughput_fullpath[Gbps]']['mean'].to_numpy()
#             exe_full_std = group['throughput_fullpath[Gbps]']['std'].to_numpy()

#             i+=1
#             if np.any(exe != 0) and F2F:
#                 series_label.append(f"U280 Seg Size {seg_size} KB  F2F")
#                 series_y.append(exe)
#                 series_x.append(bufsize)
#                 stdevs.append(exe_std)
#                 styles.append(f"C{i}-")
#             if np.any(exe_full != 0) and H2H:
#                 series_label.append(f"U280 Seg Size {seg_size} KB H2H")
#                 series_y.append(exe_full)
#                 series_x.append(bufsize)
#                 stdevs.append(exe_full_std)
#                 styles.append(f"C{i}--")

        
#         plot_lines("U280_Segment"+("H2H" if H2H else "") + ("F2F" if F2F else "")+collective.replace("/", ""), series_x, series_y, series_label, styles, y_errors=stdevs, logx=True, logy=False, throughput=True)
#         #plot_clustered_bars(collective, series_x, series_y, series_label)
            
# def compare_rank_number_and_bsize(df_accl, H2H=False, F2F=True, error = False):
#     df_accl              = df_accl[ (df_accl["rank_id"] == 0) ]
#     collectives     = df_accl["collective"].unique()
#     segment_size    = 1024
#     for collective in collectives:
#         series_label = []
#         series_y     = []
#         series_x     = []
#         styles       = []
#         stdevs       = []
#         average_delta = None
#         subset              = df_accl[ (df_accl["collective"] == collective) & 
#                                   (
#                                     ((df_accl["segment_size[B]"] == segment_size) & 
#                                      (  
#                                         ( ( df_accl["board_instance"] == "xilinx_u280_xdma_201920_3") & (df_accl["number_of_banks"] == 6)) |
#                                         ( ( df_accl["board_instance"] == "xilinx_u250_gen3x16_xdma_shell_3_1") & (df_accl["number_of_banks"] == 3))                                     
#                                      )
#                                     ) | (df_accl["board_instance"] != "OpenMPI" ))]
#         grouped             = subset.groupby(["number_of_nodes", "size[B]"]).agg({'execution_time[us]':['mean','std'], 'execution_time_fullpath[us]':['mean','std']})
#         grouped.reset_index(inplace=True)
#         print(collective, grouped)
#         nodes_available = [4,5,6,7]
#         ls = ['-', '--', '-.', ':']
#         j=0
#         for  num_nodes, sub_group in grouped.groupby(["number_of_nodes"]):
#             if num_nodes not in nodes_available:
#                 continue
#             exe          = sub_group['execution_time[us]']['mean'].to_numpy()
#             exe_std      = sub_group['execution_time[us]']['std'].to_numpy()
#             bufsize      = sub_group['size[B]'].to_numpy()*1024
#             exe_full     = sub_group['execution_time_fullpath[us]']['mean'].to_numpy()
#             exe_full_std = sub_group['execution_time_fullpath[us]']['std'].to_numpy()
            

#             if np.any(exe != 0) and F2F:
#                 series_label.append(f"ACCL {num_nodes} F2F")
#                 series_y.append(exe)
#                 series_x.append(bufsize)
#                 stdevs.append(exe_std)
#                 styles.append(f"C4{ls[j % len(ls)]}+")
#             if np.any(exe_full != 0) and H2H:
#                 series_label.append(f"ACCL {num_nodes} H2H")
#                 series_y.append(exe_full)
#                 series_x.append(bufsize)
#                 stdevs.append(exe_full_std)
#                 styles.append(f"C4{ls[j % len(ls)]}+")
#             j+=1
        
#         #OpenMPI
#         subset              = df_accl[(df_accl["collective"] == collective) & (df_accl["board_instance"] == "OpenMPI" )]
#         grouped             = subset.groupby(["number_of_nodes", "size[B]"]).agg({'execution_time[us]':['mean','std'], 'execution_time_fullpath[us]':['mean','std']})
#         grouped.reset_index(inplace=True)
#         print(grouped)

#         j=0
#         for  num_nodes, sub_group in grouped.groupby(["number_of_nodes"]):
#             if num_nodes not in nodes_available:
#                 continue
#             exe          = sub_group['execution_time[us]']['mean'].to_numpy()
#             exe_std      = sub_group['execution_time[us]']['std'].to_numpy()
#             bufsize      = sub_group['size[B]'].to_numpy()*1024
#             exe_full     = sub_group['execution_time_fullpath[us]']['mean'].to_numpy()
#             exe_full_std = sub_group['execution_time_fullpath[us]']['std'].to_numpy()
#             if np.any(exe_full != 0):
#                 if average_delta is not None and F2F:
#                     series_label.append(f"OpenMPI {num_nodes} F2F")
#                     exe = list(map(sum, zip(exe_full,average_delta)))
#                     series_y.append(exe)
#                     series_x.append(bufsize[:len(exe)])
#                     stdevs.append(None)
#                     styles.append(f"C3{ls[j % len(ls)]}+")

#                 if H2H or F2F and average_delta is None:
#                     series_label.append(f"OpenMPI {num_nodes} H2H")
#                     series_y.append(exe_full)
#                     series_x.append(bufsize)
#                     stdevs.append(exe_full_std)
#                     styles.append(f"C3{ls[j % len(ls)]}+")
#             j+=1
     
        
#         plot_lines("rank_comparison"+collective.replace("/", ""), series_x, series_y, series_label, styles, y_label='Latency [us]',legend_loc ="upper left" , y_errors=(stdevs if error else None))

#         #plot_clustered_bars(collective, series_x, series_y, series_label)

def compare_rank_with_fixed_bsize(df_accl, error=False):
    df_accl              = df_accl[ (df_accl["rank_id"] == 0) ]
    collectives     = df_accl['collective'].apply(lambda r: '_'.join(r.split('_')[:-1])).unique()
    bsizes           = df_accl[ "size[B]"].unique()
    segment_size    = 4*1024*1024
    print(collectives)
    print(bsizes)
    for collective in collectives:
        if collective != "sendrecv":
            print(collective)
            for bsize in bsizes:
                series_label = []
                series_y     = []
                series_x     = []
                styles       = []
                stdevs       = []
                subset              = df_accl[(df_accl["collective"].str.startswith(collective)) &
                                        (df_accl["size[B]"] == bsize) & 
                                        (df_accl["segment_size[B]"] == segment_size) & 
                                        (df_accl["number_of_nodes"] > 2)]
                grouped             = subset.groupby(["collective","number_of_nodes"]).agg({'execution_time[us]':['mean','std']})
                grouped.reset_index(inplace=True)
                print(collective, bsize, grouped)
                grouped             = grouped.groupby(["collective"])
                print(grouped)

                i = 0
                for coll, group in grouped:
                    print(group)
                    exe          = group['execution_time[us]']['mean'].to_numpy()
                    exe_std      = group['execution_time[us]']['std'].to_numpy()
                    num_nodes    = group['number_of_nodes'].to_numpy()
                    print(num_nodes)
                    
                    if np.any(exe != 0):
                        series_label.append(f"ACCL {coll}")
                        series_y.append(exe)
                        series_x.append(num_nodes)
                        stdevs.append(exe_std)
                        styles.append(f"C{i+1}+-")
                        i+=1

        
                # exe          = grouped['execution_time[us]']['mean'].to_numpy()
                # exe_std      = grouped['execution_time[us]']['std'].to_numpy()
                # num_nodes    = grouped['number_of_nodes']
                
                # if np.any(exe != 0):
                #     series_label.append("ACCL {}")
                #     series_y.append(exe)
                #     series_x.append(num_nodes)
                #     stdevs.append(exe_std)
                #     styles.append(f"C2-+")
                
                #OpenMPI
                # subset              = df_accl[(df_accl["collective"] == collective) & (df_accl[ "size[B]"] == bsize) & (df_accl["board_instance"].str.contains("OpenMPI") ) & (df_accl["number_of_nodes"] > 2)]
                # i=2
                # for board_name, group in subset.groupby(["board_instance"]): 
                #     grouped             = group.groupby(["number_of_nodes"]).agg({'execution_time[us]':['mean','std'], 'execution_time_fullpath[us]':['mean','std']})
                #     grouped.reset_index(inplace=True)

        
                #     exe          = grouped['execution_time[us]']['mean'].to_numpy()
                #     exe_std      = grouped['execution_time[us]']['std'].to_numpy()
                #     num_nodes    = grouped['number_of_nodes'].to_numpy()
                #     exe_full     = grouped['execution_time_fullpath[us]']['mean'].to_numpy()
                #     exe_full_std = grouped['execution_time_fullpath[us]']['std'].to_numpy()
                #     i+=1
                #     if np.any(exe != 0) and F2F:
                #         series_label.append(f"{board_name}")
                #         series_y.append(exe)
                #         series_x.append(num_nodes[:len(exe)])
                #         stdevs.append(exe_std)
                #         styles.append(f"C{i}-+")

                #     if np.any(exe_full) and H2H:
                #             series_label.append(f"{board_name} H2H")
                #             series_y.append(exe_full)
                #             series_x.append(num_nodes)
                #             stdevs.append(exe_full_std)
                #             styles.append(f"C{i}--+")

                
                plot_lines("rank_comparison_"+collective.replace("/", "")+"_"+str(bsize), series_x, series_y, series_label, styles, x_label="Number of ranks", y_label='Latency [us]', legend_loc ="upper left", logx=False, logy = False, y_errors=(stdevs if error else None))

                #plot_clustered_bars(collective, series_x, series_y, series_label)

# def simplify_board_name(name):
#     if   name == "xilinx_u250_gen3x16_xdma_shell_3_1":
#         return "U250"
#     elif name == "xilinx_u280_xdma_201920_3":
#         return "U280"
#     else:
#         return name



# def segment_vs_membank(df_accl):
#     df_accl              = df_accl[ (df_accl["experiment"] == "u280_sendrecv") & (df_accl["rank_id"] == 0)  & ( df_accl["collective"] == "Send/recv")]
    
#     for board in ["xilinx_u280_xdma_201920_3" , "xilinx_u250_gen3x16_xdma_shell_3_1"]:
#         data_to_be_plotted  = []
#         subset          = df_accl[(df_accl["board_instance"] == board)]
#         if subset["number_of_banks"].count() == 0:
#             continue
#         max_banks       = subset["number_of_banks"].max()-1
#         min_banks       = subset["number_of_banks"].min()-1

        
#         banks           = list(range(min_banks,max_banks+1))
#         segment_sizes   = list(sorted(subset["segment_size[B]"].unique()))
#         grouped         = subset.groupby(["number_of_banks", "segment_size[B]", "size[B]"]).agg({'throughput[Gbps]':['mean','std'], 'throughput_fullpath[Gbps]':['mean','std']})
        
#         for _ in banks:
#             data_to_be_plotted.append([ 0 for _ in segment_sizes])

#         grouped.reset_index(inplace=True)
#         for ((curr_num_banks, curr_segment ),  group) in grouped.groupby(["number_of_banks", "segment_size[B]"]):
#             thr          = group['throughput[Gbps]']['mean'].max()
#             print(curr_num_banks, curr_segment, thr)

#             data_to_be_plotted[curr_num_banks-1-min_banks][np.argwhere(segment_sizes == curr_segment)[0][0]] = thr
        


#         fig, ax = plt.subplots()
#         im = ax.imshow(data_to_be_plotted, cmap="Wistia" )
#         #im = ax.imshow(data_to_be_plotted, cmap="Wistia", vmin=0, vmax=100 )
#         ax.invert_yaxis()
#         #ax.invert_xaxis()
#         cbar = ax.figure.colorbar(im, ax=ax)
#         cbar.ax.tick_params(labelsize=14)
#         # We want to show all ticks...
#         ax.set_yticks(np.arange(len(banks)))
#         ax.set_xticks(np.arange(len(segment_sizes)))
#         # ... and label them with the respective list entries
#         ax.set_yticklabels(banks,           fontsize=14)
#         ax.set_xticklabels(segment_sizes,   fontsize=14)
#         ax.set_ylabel("number_of_banks",    fontsize=16)
#         ax.set_xlabel("Segment size [KB]",  fontsize=16)
#         # Loop over data dimensions and create text annotations.
#         for i in range(len(banks)):
#             for j in range(len(segment_sizes)):
#                 if data_to_be_plotted[i][j] > 0:
#                     d = data_to_be_plotted[i][j]
#                     ax.text(j, i, f"{d:.1f}", ha="center", va="center", fontsize=14)
        
#         ax.set_title("Peak throughput [Gbps]", fontsize=16)
#         plt.show()
#         plt.savefig(f"segment_vs_bank_{board}.png", format='png', bbox_inches='tight')

# def optimized_vs_base(df_accl, selection_params,  error = False, logy=False):
#     df_accl              = df_accl[ (df_accl["rank_id"] == 0) ]

#     series_label = []
#     series_y     = []
#     series_x     = []
#     styles       = []
#     stdevs       = []
#     i=0
#     collectives  = []
#     for selection_param in selection_params:
#         exp             = selection_param["experiment"]
#         label           = selection_param["label"]
#         collective_name = selection_param["collective"]
#         seg_size        = selection_param["segment_size[B]"]
#         board           = selection_param["board_instance"]
#         num_banks       = selection_param["number_of_banks"]
#         num_nodes       = selection_param["number_of_nodes"]
#         F2F             = selection_param["F2F"]
#         H2H             = selection_param["H2H"]
#         collectives.append(collective_name.replace("/", ""))

#         subset              = df_accl[(df_accl["experiment"]      == exp)             &
#                                  (df_accl["collective"] == collective_name) & (df_accl["segment_size[B]"]   == seg_size ) &
#                                  (df_accl["board_instance"]  == board)           & (df_accl["number_of_banks"]    == num_banks) &
#                                  (df_accl["number_of_nodes"] == num_nodes)]
#         grouped             = subset.groupby(["size[B]"]).agg({'execution_time[us]':['mean','std'], 'execution_time_fullpath[us]':['mean','std']})
#         grouped.reset_index(inplace=True)
#         print(grouped)
        

#         exe          = grouped['execution_time[us]']['mean'].to_numpy()
#         exe_std      = grouped['execution_time[us]']['std'].to_numpy()
#         bufsize      = grouped['size[B]'].to_numpy()*1024
#         exe_full     = grouped['execution_time_fullpath[us]']['mean'].to_numpy()
#         exe_full_std = grouped['execution_time_fullpath[us]']['std'].to_numpy()

#         board = simplify_board_name(board)
#         if np.any(exe != 0) and F2F:
#             series_label.append(f"{label} F2F")
#             series_y.append(exe)
#             series_x.append(bufsize)
#             stdevs.append(exe_std)
#             styles.append(f"C{i}-")
#             i+=1
#         if np.any(exe_full != 0) and H2H:
#             series_label.append(f"{label} H2H")
#             series_y.append(exe_full)
#             series_x.append(bufsize)
#             stdevs.append(exe_full_std)
#             styles.append(f"C{i}-")
#             i+=1

#     #optimized version

#     plot_lines("comparison"+"_".join(collectives), series_x, series_y, series_label, styles, y_label='Latency [us]', logx=True, logy=logy, legend_loc ="upper left", y_errors=(stdevs if error else None))
        
def remove_multiple_headers(df):
    headers = df.columns.tolist()
    df = df[df[headers[0]]!=headers[0]].reset_index(drop=True)
    for column_name in ["number of nodes","rank id","number of banks","buffer size[KB]","segment_size[KB]","execution_time[us]","throughput[Gbps]","execution_time_fullpath[us]","throughput_fullpath[Gbps]"]:
        df[column_name] = pd.to_numeric(df[column_name])
    return df

def load_csvs_under(path):

    csv_files    = [join(path, f) for f in listdir(path)  if (isfile(join(path, f)) and f.find(".csv") != -1)]
    print("csv files ingested", csv_files)
    csvs = []
    for csv_path in csv_files:
        csvs.append(pd.read_csv(csv_path))
    return pd.concat(csvs)

def load_logs_under(path):

    csv_files    = [join(path, f) for f in listdir(path)  if (isfile(join(path, f)) and f.find(".log") != -1)]
    print("csv files ingested", csv_files)
    csvs = []
    for csv_path in csv_files:
        csvs.append(pd.read_csv(csv_path))
    return pd.concat(csvs)

def compare_throughput(df_accl, df_mpi):
    df_accl              = df_accl[ (df_accl["rank_id"] == 0)  & ( (df_accl["collective"] == "sendrecv_H2H") | (df_accl["collective"] == "sendrecv_F2F") | (df_accl["collective"] == "sendrecv_K2K"))]
    print(df_accl)
    segment_size    = 4194304
    series_label = []
    series_y     = []
    series_x     = []
    styles       = []
    stdevs       = []

    subset              = df_accl[(df_accl["segment_size[B]"] == segment_size) & (df_accl["number_of_banks"] == 6)]
    grouped             = subset.groupby(["collective","size[B]"]).agg({'throughput[Gbps]':['mean','std']})
    grouped.reset_index(inplace=True)
    print(grouped)

    grouped             = grouped.groupby(["collective"])
    print(grouped)
    j=0
    for i,(collective,group) in enumerate(grouped):
        print(group)
        thr          = group['throughput[Gbps]']['mean'].to_numpy()
        thr_std      = group['throughput[Gbps]']['std'].to_numpy()
        bufsize      = group['size[B]'].to_numpy()
        if np.any(thr != 0):
            series_label.append(f"ACCL {collective}")
            series_y.append(thr)
            series_x.append(bufsize)
            stdevs.append(thr_std)
            styles.append(f"C{j+1}-+")
            j+=1
        
    #OpenMPI
    subset              = df_mpi[( (df_mpi["rank id"] == 0) & (df_mpi["board_instance"] == "OpenMPI" ) & ( df_mpi["collective name"] == "Send/recv") )]
    grouped             = subset.groupby(["buffer size[KB]"]).agg({'throughput[Gbps]':['mean','std'], 'throughput_fullpath[Gbps]':['mean','std']})
    grouped.reset_index(inplace=True)

    print(grouped)
    thr          = grouped['throughput[Gbps]']['mean'].to_numpy()
    thr_std      = grouped['throughput[Gbps]']['std'].to_numpy()
    bufsize      = grouped['buffer size[KB]'].to_numpy()*1024
    thr_full     = grouped['throughput_fullpath[Gbps]']['mean'].to_numpy()
    thr_full_std = grouped['throughput_fullpath[Gbps]']['std'].to_numpy()
    #if np.any(thr != 0) and F2F:
    #        series_label.append("OpenMPI F2F")
    #        series_y.append(thr)
    #        series_x.append(bufsize[:len(thr)])
    #        stdevs.append(thr_std)
    #        styles.append(f"C3-+")
    #if np.any(thr_full != 0) and H2H :
    if np.any(thr_full != 0) :
            series_label.append("OpenMPI_H2H")
            series_y.append(thr_full)
            series_x.append(bufsize)
            stdevs.append(thr_full_std)
            styles.append(f"C{j+1}-+")
            j+=1

        
    plot_lines("sendrecv_throughput_comparsion", series_x, series_y, series_label, styles, x_label="Message Size", y_label='Throughput [Gbps]', legend_loc ="upper left", logx=True, logy = False)




if __name__ == "__main__":
    accl_path            ="./"
    mpi_path             ="./open_mpi_log"
    df_accl = load_logs_under(accl_path)
    df_mpi=load_csvs_under(mpi_path)

    df_mpi = remove_multiple_headers(df_mpi)

    import argparse

    parser = argparse.ArgumentParser(description='Creates some graphs.')
    # parser.add_argument('--statistic'           , action='store_true', default=True,     help='count runs of experiments'                     )
    # parser.add_argument('--sendrecv_banks'      , action='store_true', default=False,     help='send recv throughput with different number_of_banks'                     )
    # parser.add_argument('--sendrecv_seg'        , action='store_true', default=False,     help='send recv throughput with different number of segmentation size'                     )
    # parser.add_argument('--norm'                , action='store_true', default=False,     help='test normality'                          )
    # parser.add_argument('--ssize'               , action='store_true', default=False,     help='ssize vs buffer size'                     )
    # parser.add_argument('--board'               , action='store_true', default=False,     help='compare performance of different alveo'   )
    # parser.add_argument('--openMPI'             , action='store_true', default=False,     help='compare performance against OpenMPI'   )
    parser.add_argument('--rank'                , action='store_true', default=False,    help='compare performance of different number of ranks'   )
    # parser.add_argument('--rank2_number'        , action='store_true', default=False,     help='compare performance of different number of ranks'   )
    parser.add_argument('--throughput'          , action='store_true', default=False,     help='compare throughput'   )
    parser.add_argument('--latency'             , action='store_true', default=False,     help='compare latency'   )
    # parser.add_argument('--segment_vs_membank'  , action='store_true', default=False,     help='compare throughput changing segment size and number of memory banks used'   )
    # parser.add_argument('--optimized_vs_base'   , action='store_true', default=False,     help='comapre execution time of bcast when optimized'   )
    

    #                                                                                                                       fig           data  
    #D)for every collectiveopen_mpi_and_fpga_at_different_ranks (probably 4-8 with fpga not full_path)                      6xfig         ok    
    #Z)send and receive throughput with different banks (1)                                                                 1xfig         ok
    #Z)send and receive throughput with different segment size (2)                                                          1             ko  
    #Z)send and receive latency    with different banks (1)                 (needed?)                                                     ok
    #Z)send and receive latency    with different segment size (2)          (needed?)                                                     ko   
    #Z)show throughput send-recv and compare with openmpi                                                                   1xfig         ok
    #optimizations 
    #D)   bcast scatter   : one of them                                                                                                   ok(u280 dual datapath baseline/ u280 rr)  
    #D)   reduce          : avoid store intermediate results (reduce UXX vs reduce UXX dual datapath. Shows streaming kernels)  x)        ok(upto 16 MB, dualpath / u280 baseline)
    #D)   allreduce       : can be overlapped with reduce                                                                       x)        ok(upto 32 MB, dualpath / u280 baseline, similar to reduce(we don't show naive implementation), difference w.r.t reduce would be more pronounced)
    # WE need to rename the experiments (data)
    #   different algorithm(scatter and bcast rr/none)
    # the other u280 baseline might be better than dual datapath (waiting for packetizer)

    args = parser.parse_args()
    # if args.norm:
    #     normality_test(df_accl)
    # if args.ssize:
    #     compare_ssize(df_accl)
    # if args.board:
    #     compare_board(df_accl)
    if args.rank:
        compare_rank_with_fixed_bsize(df_accl)
    # if args.rank2_number:
    #     compare_rank_with_fixed_bsize(df_accl, error=True)
    # if args.statistic:
    #     get_statistics(df_accl)
    # if args.openMPI:
    #     compare_openMPI(df_accl)
    #     #compare_openMPI(df_accl, H2H=False)
    #     #compare_openMPI(df_accl, F2F=False)
    #     #compare_box_plot(df_accl)
    #     compare_box_plot_with_fixed_bsize(df_accl)
    if args.throughput:
        compare_throughput(df_accl, df_mpi)
    if args.latency:
        compare_latency(df_accl, df_mpi)
    # if args.sendrecv_banks:
    #     sendrecv_banks(df_accl)
    # if args.sendrecv_seg:
    #     sendrecv_segmentation(df_accl)
    # if args.segment_vs_membank:
    #     segment_vs_membank(df_accl)
    # if args.optimized_vs_base:
    #     other = pd.concat([df_accl, load_csvs_under("accl/dual_path")])
    #     other = remove_multiple_headers(other)

    #     optimized_vs_base(other,[{ "experiment":"bcast",
    #                             "label":"baseline: dual datapath",
    #                             "collective":"Broadcast",
    #                             "segment_size[B]":1024,
    #                             "board_instance":"xilinx_u280_xdma_201920_3", 
    #                             "number_of_banks":6,
    #                             "number_of_nodes": 4,
    #                             "F2F":True,
    #                             "H2H":False},
    #                             { "experiment":"bcast1",
    #                             "label":"baseline: dual datapath 2",
    #                             "collective":"Broadcast",
    #                             "segment_size[B]":1024,
    #                             "board_instance":"xilinx_u280_xdma_201920_3", 
    #                             "number_of_banks":6,
    #                             "number_of_nodes": 4,
    #                             "F2F":True,
    #                             "H2H":False}, 
    #                             { "experiment":"bcast_rr",
    #                             "label":"baseline: dual datapath rr",
    #                             "collective":"Broadcast",
    #                             "segment_size[B]":1024,
    #                             "board_instance":"xilinx_u280_xdma_201920_3", 
    #                             "number_of_banks":6,
    #                             "number_of_nodes": 4,
    #                             "F2F":True,
    #                             "H2H":False},
    #                             { "experiment":"u280_bcast",
    #                             "label":"single datapath: bcast rr",
    #                             "collective":"Broadcast",
    #                             "segment_size[B]":1024,
    #                             "board_instance":"xilinx_u280_xdma_201920_3", 
    #                             "number_of_banks":6,
    #                             "number_of_nodes": 4,
    #                             "F2F":True,
    #                             "H2H":False}, 
    #                             { "experiment":"converted from log",
    #                             "label":"OpenMPI",
    #                             "collective":"Broadcast",
    #                             "segment_size[B]":0,
    #                             "board_instance":"OpenMPI", 
    #                             "number_of_banks": 0,
    #                             "number_of_nodes": 4,
    #                             "F2F":False,
    #                             "H2H":True}], logy=True)

    #     optimized_vs_base(other,[{ "experiment":"u280_dual_path_scatter",
    #                             "label":"baseline",
    #                             "collective":"Scatter",
    #                             "segment_size[B]":1024,
    #                             "board_instance":"xilinx_u280_xdma_201920_3", 
    #                             "number_of_banks":6,
    #                             "number_of_nodes": 4,
    #                             "F2F":True,
    #                             "H2H":False}, 
    #                             { "experiment":"u280_scatter",
    #                             "label":"alternative",
    #                             "collective":"Scatter",
    #                             "segment_size[B]":1024,
    #                             "board_instance":"xilinx_u280_xdma_201920_3", 
    #                             "number_of_banks":6,
    #                             "number_of_nodes": 4,
    #                             "F2F":True,
    #                             "H2H":False}])

    #     optimized_vs_base(other,[{ "experiment":"u280_dual_path_reduce",
    #                             "label":"baseline",
    #                             "collective":"Reduce",
    #                             "segment_size[B]":1024,
    #                             "board_instance":"xilinx_u280_xdma_201920_3", 
    #                             "number_of_banks":6,
    #                             "number_of_nodes": 4,
    #                             "F2F":True,
    #                             "H2H":False}, 
    #                             { "experiment":"u280_reduce",
    #                             "label":"alternative",
    #                             "collective":"Reduce",
    #                             "segment_size[B]":1024,
    #                             "board_instance":"xilinx_u280_xdma_201920_3", 
    #                             "number_of_banks":6,
    #                             "number_of_nodes": 4,
    #                             "F2F":True,
    #                             "H2H":False}])