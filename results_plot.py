import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib import colors as mcolors


def plot_opt_processes(files_1: list, files_2: list, legends_1: list, legends_2: list, name="compare_fixed",
                       data_file="losses-rand",
                       y_ticks=[1e1, 1e-2, 1e-4, 1e-6, 1e-7]):
    plt.rcParams['axes.linewidth'] = 0.5
    myfont = {'size': 22, 'family': 'Helvetica'}
    myfont_legend = {'size': 14, 'family': 'Helvetica'}
    plt.rcParams["axes.edgecolor"] = "black"

    # palette = sns.color_palette()
    # print(palette)
    fg = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    markers1 = ['*', '.']
    markers2 = ['1', '2']
    # 3:
    colors = ['b', 'tab:orange', 'g', 'y', 'm', 'tab:brown',
              'tab:pink', 'c', 'r', 'tab:olive', 'C0']

    # color_map = ['b', 'tab:orange', 'g', 'r', 'm', 'tab:brown', 'gold', 'forestgreen',
    #              'tab:pink', 'c', 'y', 'tab:olive', 'C0', 'slategray', 'indigo', 'crimson', 'fuchsia']
    # colors = iter(color_map)

    _len = 1000
    # if not is_ood else 200
    #  if not is_ood else [1e2, 1e-2, 1e-4, 1e-6, 1e-7]
    x_ticks = [1e0, 1e1, 1e2, 1e3]
    # if not is_ood else [1e5, 1e4, 1e3, 1e2]
    # data_file = "losses-rand" if not is_ood else "losses-rand-OOD"
    for i in range(len(legends_1)):
        y = np.loadtxt("results/" + files_1[i] + "/" + data_file)
        y = y[:_len]
        # sns.lineplot(data=y, palette=palette[i])
        # mean_loss = np.mean(y)
        c = colors[i]
        #  + '\t' + '{:.2e}'.format(mean_loss)
        plt.plot(y, label=legends_1[i], linewidth=1,
                 alpha=0.8, color=c, linestyle='dashed', marker=markers1[0],
                 markerfacecolor=c, markersize=6, markevery=0.1)

    if files_2 is not None:
        # colors = iter(color_map)
        for i in range(len(legends_2)):
            y = np.loadtxt("results/" + files_2[i] + "/" + data_file)
            y = y[:_len]
            mean_loss = np.mean(y)
            c = colors[i]
            #  + ' ' + '{:.2e}'.format(mean_loss)
            plt.plot(y, label=legends_2[i], linewidth=1,
                     alpha=0.8, color=c, linestyle='-', marker=markers1[1],
                     markerfacecolor=c, markersize=7, markevery=0.1)

    sns.set_style("whitegrid", {"axes.edgecolor": "black"})

    plt.legend(loc='upper right',
               #    bbox_to_anchor=(0, 1.05),
               prop=myfont_legend,
               #    ncols=4,

               frameon=1, framealpha=0.5)
    ax.set_ylim(0.94*y_ticks[-1], 1.1*y_ticks[0])
    # if not is_ood:
    #     ax.set_ylim(10**(-8), 10**1)
    # else:
    #     ax.set_ylim(10**(-8), 10**4)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.grid(which='major', axis='both', linestyle='dashed')
    plt.ylabel('$(F(x_k) - F(x^*))/F(x^*)$', fontdict=myfont)
    plt.xlabel('Iteration $k$', fontdict=myfont)
    plt.xticks(x_ticks, fontsize=myfont['size'] // 3 * 2)
    plt.yticks(y_ticks, fontsize=myfont['size'] // 3 * 2)
    plt.tight_layout()
    plt.savefig("plots/" + name + ".pdf")


def plot_opt_processes_real(files_1: list, files_2: list, legends_1: list, legends_2: list, name="compare_fixed",
                            data_file="losses-rand",
                            y_ticks=[1e1, 1e-2, 1e-4, 1e-6, 1e-7]):
    plt.rcParams['axes.linewidth'] = 0.5
    myfont = {'size': 22, 'family': 'Helvetica'}
    myfont_legend = {'size': 14, 'family': 'Helvetica'}
    plt.rcParams["axes.edgecolor"] = "black"

    # palette = sns.color_palette()
    # print(palette)
    fg = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    markers1 = ['*', '.']
    # markers2 = ['1', '2']
    # 3: 'g',
    colors = ['b', 'tab:orange', 'y', 'm', 'tab:brown',
              'tab:pink', 'c', 'r', 'tab:olive', 'C0']

    # color_map = ['b', 'tab:orange', 'g', 'r', 'm', 'tab:brown', 'gold', 'forestgreen',
    #              'tab:pink', 'c', 'y', 'tab:olive', 'C0', 'slategray', 'indigo', 'crimson', 'fuchsia']
    # colors = iter(color_map)

    _len = 1000
    # if not is_ood else 200
    #  if not is_ood else [1e2, 1e-2, 1e-4, 1e-6, 1e-7]
    x_ticks = [1e0, 1e1, 1e2, 1e3]
    # if not is_ood else [1e5, 1e4, 1e3, 1e2]
    # data_file = "losses-rand" if not is_ood else "losses-rand-OOD"
    for i in range(len(legends_1)):
        y = np.loadtxt("results/" + files_1[i] + "/" + data_file)
        y = y[:_len]
        # sns.lineplot(data=y, palette=palette[i])
        mean_loss = np.mean(y)
        c = colors[i]
        #  + '\t' + '{:.2e}'.format(mean_loss)
        plt.plot(y, label=legends_1[i], linewidth=1,
                 alpha=0.8, color=c, linestyle='dashed', marker=markers1[0],
                 markerfacecolor=c, markersize=6, markevery=0.1)

    if files_2 is not None:
        # colors = iter(color_map)
        for i in range(len(legends_2)):
            y = np.loadtxt("results/" + files_2[i] + "/" + data_file)
            y = y[:_len]
            mean_loss = np.mean(y)
            c = colors[i]
            #  + ' ' + '{:.2e}'.format(mean_loss)
            plt.plot(y, label=legends_2[i], linewidth=1,
                     alpha=0.8, color=c, linestyle='-', marker=markers1[1],
                     markerfacecolor=c, markersize=7, markevery=0.1)

    sns.set_style("whitegrid", {"axes.edgecolor": "black"})

    plt.legend(loc='upper right',
               #    bbox_to_anchor=(0, 1.05),
               prop=myfont_legend,
               #    ncols=4,

               frameon=1, framealpha=0.5)
    ax.set_ylim(0.94*y_ticks[-1], 1.1*y_ticks[0])
    # if not is_ood:
    #     ax.set_ylim(10**(-8), 10**1)
    # else:
    #     ax.set_ylim(10**(-8), 10**4)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.grid(which='major', axis='both', linestyle='dashed')
    plt.ylabel('$(F(x_k) - F(x^*))/F(x^*)$', fontdict=myfont)
    plt.xlabel('Iteration $k$', fontdict=myfont)
    plt.xticks(x_ticks, fontsize=myfont['size'] // 3 * 2)
    plt.yticks(y_ticks, fontsize=myfont['size'] // 3 * 2)
    plt.tight_layout()
    plt.savefig("plots/" + name + ".pdf")


def plot_ablation_ood_processes(files_1: list, files_2: list,
                                legends_1: list, legends_2: list,
                                name="compare_fixed",
                                y_ticks=[1e3, 1e-2, 1e-4, 1e-6, 1e-7],
                                _loc='upper right',
                                is_y_log=True):
    plt.rcParams['axes.linewidth'] = 0.5
    myfont = {'size': 22, 'family': 'Helvetica'}
    myfont_legend = {'size': 10, 'family': 'Helvetica'}
    plt.rcParams["axes.edgecolor"] = "black"

    # palette = sns.color_palette()
    # print(palette)
    fg = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    markers1 = ['*', '.']
    markers2 = ['1', '2']
    colors = ['b', 'tab:orange', 'g', 'y', 'm', 'tab:brown',
              'tab:pink', 'c', 'r', 'tab:olive', 'C0', 'forestgreen']

    # color_map = ['b', 'tab:orange', 'g', 'r', 'm', 'tab:brown', 'gold', 'forestgreen',
    #              'tab:pink', 'c', 'y', 'tab:olive', 'C0', 'slategray', 'indigo', 'crimson', 'fuchsia']
    # colors = iter(color_map)

    _len = 1000
    # if not is_ood else 200
    #  if not is_ood else [1e2, 1e-2, 1e-4, 1e-6, 1e-7]
    x_ticks = [1e0, 1e1, 1e2, 1e3]
    # if not is_ood else [1e5, 1e4, 1e3, 1e2]
    # data_file = "losses-rand" if not is_ood else "losses-rand-OOD"
    for i in range(len(files_1)):
        y = np.loadtxt("results/" + files_1[i])
        y = y[:_len]
        # sns.lineplot(data=y, palette=palette[i])
        # mean_loss = np.mean(y)
        c = colors[i]
        plt.plot(y, label=legends_1[i], linewidth=1,
                 alpha=0.8, color=c, linestyle='dashed', marker=markers1[0],
                 markerfacecolor=c, markersize=6, markevery=0.1)

    # colors = iter(color_map)
    for i in range(len(files_2)):
        y = np.loadtxt("results/" + files_2[i])
        y = y[:_len]
        # mean_loss = np.mean(y)
        c = colors[i]
        plt.plot(y, label=legends_2[i], linewidth=1,
                 alpha=0.8, color=c, linestyle='-', marker=markers1[1],
                 markerfacecolor=c, markersize=7, markevery=0.1)

    sns.set_style("whitegrid", {"axes.edgecolor": "black"})

    plt.legend(loc=_loc,
               #    bbox_to_anchor=(0, 1.05),
               prop=myfont_legend,
               #    ncols=4,

               frameon=1, framealpha=0.5)
    # if not is_ood:
    #     ax.set_ylim(10**(-8), 10**1)
    # else:
    #     ax.set_ylim(10**(-8), 10**4)
    if is_y_log:
        ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(0.94*y_ticks[-1], 1.1*y_ticks[0])
    plt.grid(which='major', axis='both', linestyle='dashed')
    plt.ylabel('$(F(x_k) - F(x^*))/F(x^*)$', fontdict=myfont)
    plt.xlabel('Iteration $k$', fontdict=myfont)
    plt.xticks(x_ticks, fontsize=myfont['size'] // 3 * 2)
    plt.yticks(y_ticks, fontsize=myfont['size'] // 3 * 2)
    plt.tight_layout()
    plt.savefig("plots/" + name + ".pdf")


def plot_training_config_opt_processes(files_1: list, legends_1: list, name="compare_fixed",
                                       data_file="losses-rand",
                                       _loc='upper right',
                                       y_ticks=[1e1, 1e-2, 1e-4, 1e-6, 1e-7]):
    plt.rcParams['axes.linewidth'] = 0.5
    myfont = {'size': 22, 'family': 'Helvetica'}
    myfont_legend = {'size': 14, 'family': 'Helvetica'}
    plt.rcParams["axes.edgecolor"] = "black"

    # palette = sns.color_palette()
    # print(palette)
    fg = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    markers1 = ['.']
    markers = [
        'o',  # Circle
        's',  # Square
        '^',  # Triangle up
        'v',  # Triangle down
        '<',  # Triangle left
        '>',  # Triangle right
        'p',  # Pentagon
        'P',  # Plus (filled)
        '*',  # Star
        'h',  # Hexagon
        'H',  # Hexagon2
        'X',  # X (filled)
        'D',  # Diamond
        'd',  # Thin diamond
        '|',  # Vline
        '_',  # Hline
        '4'
    ]
    # markers2 = ['1', '2']
    # 3:
    # colors = ['b', 'tab:orange', 'g', 'y', 'm', 'tab:brown',
    #           'tab:pink', 'c', 'r', 'tab:olive', 'C0']

    # colors = ['b', 'tab:orange', 'g', 'r', 'm', 'tab:brown', 'gold', 'forestgreen',
    #           'tab:pink', 'c', 'y', 'tab:olive', 'C0', 'slategray', 'indigo', 'crimson', 'fuchsia']
    colors = [
        '#e6194B',  # Red
        '#3cb44b',  # Green
        '#ffe119',  # Yellow
        '#4363d8',  # Blue
        '#f58231',  # Orange
        '#911eb4',  # Purple
        '#46f0f0',  # Cyan
        '#f032e6',  # Magenta
        '#bcf60c',  # Lime
        '#fabebe',  # Pink
        '#008080',  # Teal
        '#e6beff',  # Lavender
        '#9a6324',  # Brown
        '#fffac8',  # Beige
        '#800000',  # Maroon
        '#aaffc3',  # Mint
        '#006400'
    ]
    # colors = iter(color_map)

    _len = 1000
    # if not is_ood else 200
    #  if not is_ood else [1e2, 1e-2, 1e-4, 1e-6, 1e-7]
    x_ticks = [1e0, 1e1, 1e2, 1e3]
    # if not is_ood else [1e5, 1e4, 1e3, 1e2]
    # data_file = "losses-rand" if not is_ood else "losses-rand-OOD"
    for i in range(len(legends_1)):
        y = np.loadtxt("results/" + files_1[i] + "/" + data_file)
        y = y[:_len]
        # sns.lineplot(data=y, palette=palette[i])
        mean_loss = np.mean(y)
        c = colors[i]
        #  + '\t' + '{:.2e}'.format(mean_loss)
        plt.plot(y, label=legends_1[i], linewidth=1,
                 alpha=0.8, color=c, linestyle='dashed', marker=markers[i],
                 markerfacecolor=c, markersize=3, markevery=0.1)

    sns.set_style("whitegrid", {"axes.edgecolor": "black"})

    plt.legend(loc=_loc,
               #    bbox_to_anchor=(0, 1.05),
               prop=myfont_legend,
               #    ncols=4,
               frameon=1, framealpha=0.5)
    ax.set_ylim(0.94*y_ticks[-1], 1.1*y_ticks[0])
    # if not is_ood:
    #     ax.set_ylim(10**(-8), 10**1)
    # else:
    #     ax.set_ylim(10**(-8), 10**4)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.grid(which='major', axis='both', linestyle='dashed')
    plt.ylabel('$(F(x_k) - F(x^*))/F(x^*)$', fontdict=myfont)
    plt.xlabel('Iteration $k$', fontdict=myfont)
    plt.xticks(x_ticks, fontsize=myfont['size'] // 3 * 2)
    plt.yticks(y_ticks, fontsize=myfont['size'] // 3 * 2)
    plt.tight_layout()
    plt.savefig("plots/" + name + ".pdf")


def plot_model_config_opt_processes(files_1: list, legends_1: list, name="compare_fixed",
                                    data_file="losses-rand",
                                    _loc='upper right',
                                    y_ticks=[1e1, 1e-2, 1e-4, 1e-6, 1e-7]):
    plt.rcParams['axes.linewidth'] = 0.5
    myfont = {'size': 22, 'family': 'Helvetica'}
    myfont_legend = {'size': 7, 'family': 'Helvetica'}
    plt.rcParams["axes.edgecolor"] = "black"

    # palette = sns.color_palette()
    # print(palette)
    fg = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    markers = [
        'o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd', '|', '_',
        (3, 0, 0), (4, 0, 0), (5, 0, 20), (6, 0, 0), (7,
                                                      0, 0), (8, 0, 30), (9, 0, 0), (10, 0, 0),
        '1', '2', '3', '4'
    ]

    colors = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
        '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
        '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000', '#a9a9a9', '#0000FF',
        '#fa8072', '#800080', '#fae7b5', '#006400'
    ]
    # colors = iter(color_map)

    _len = 1000
    # if not is_ood else 200
    #  if not is_ood else [1e2, 1e-2, 1e-4, 1e-6, 1e-7]
    x_ticks = [1e0, 1e1, 1e2, 1e3]
    # if not is_ood else [1e5, 1e4, 1e3, 1e2]
    # data_file = "losses-rand" if not is_ood else "losses-rand-OOD"
    for i in range(len(legends_1)):
        y = np.loadtxt("results/" + files_1[i] + "/" + data_file)
        y = y[:_len]
        # sns.lineplot(data=y, palette=palette[i])
        mean_loss = np.mean(y)
        c = colors[i]
        #  + '\t' + '{:.2e}'.format(mean_loss)
        plt.plot(y, label=legends_1[i], linewidth=1,
                 alpha=0.8, color=c, linestyle='dashed', marker=markers[i],
                 markerfacecolor=c, markersize=3, markevery=0.1 + i*0.01)

    sns.set_style("whitegrid", {"axes.edgecolor": "black"})

    plt.legend(loc=_loc,
               #    bbox_to_anchor=(0, 1.05),
               prop=myfont_legend,
               #    ncols=4,
               frameon=1, framealpha=0.5)
    ax.set_ylim(0.94*y_ticks[-1], 1.1*y_ticks[0])
    # if not is_ood:
    #     ax.set_ylim(10**(-8), 10**1)
    # else:
    #     ax.set_ylim(10**(-8), 10**4)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.grid(which='major', axis='both', linestyle='dashed')
    plt.ylabel('$(F(x_k) - F(x^*))/F(x^*)$', fontdict=myfont)
    plt.xlabel('Iteration $k$', fontdict=myfont)
    plt.xticks(x_ticks, fontsize=myfont['size'] // 3 * 2)
    plt.yticks(y_ticks, fontsize=myfont['size'] // 3 * 2)
    plt.tight_layout()
    plt.savefig("plots/" + name + ".pdf")


def debug_plot_opt_processes(legends_1: list, legends_2: list, name="compare_fixed", is_ood=False, data_file="losses-rand"):
    plt.rcParams['axes.linewidth'] = 0.5
    myfont = {'size': 22, 'family': 'Helvetica'}
    myfont_legend = {'size': 6, 'family': 'Helvetica'}
    plt.rcParams["axes.edgecolor"] = "black"

    # palette = sns.color_palette()
    # print(palette)
    fg = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    markers1 = ['*', '.']
    markers2 = ['1', '2']
    # colors = ['b', 'tab:orange', 'g', 'r', 'm', 'tab:brown',
    #           'tab:pink', 'c', 'y', 'tab:olive', 'C0']

    # color_map = ['b', 'tab:orange', 'g', 'r', 'm', 'tab:brown', 'gold', 'forestgreen',
    #              'tab:pink', 'c', 'y', 'tab:olive', 'C0', 'slategray', 'indigo', 'crimson', 'fuchsia']
    # colors = iter(color_map)

    line_num = max(len(legends_1), len(legends_2)
                   ) if legends_2 else len(legends_1)
    hsv = matplotlib.colormaps.get_cmap('hsv')
    colors = iter(hsv(np.linspace(0, 0.9, line_num)))

    # colors = iter(cm.get_cmap('hsv', np.linspace(1, line_num, line_num)))
    # colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    # colors = list(mcolors._colors_full_map.values())

    _len = 10
    # if not is_ood else 200
    #  if not is_ood else [1e2, 1e-2, 1e-4, 1e-6, 1e-7]
    y_ticks = [1e0, 1e-2, 1e-4, 1e-6, 1e-7]
    # if not is_ood else [1e5, 1e4, 1e3, 1e2]
    # plt.hsv()
    # data_file = "losses-rand" if not is_ood else "losses-rand-OOD"
    for i in range(len(legends_1)):
        _legend = legends_1[i]
        try:
            y = np.loadtxt("results/" + _legend + "/" + data_file)
        except:
            continue
        y = y[:_len]
        # sns.lineplot(data=y, palette=palette[i])
        mean_loss = np.mean(y)
        c = next(colors)
        plt.plot(y, label=_legend + ' ' + '{:.4e}'.format(mean_loss), linewidth=1,
                 alpha=0.8, color=c, linestyle='--', marker=markers1[i % 2],
                 markerfacecolor=c, markersize=6, markevery=5)

    if legends_2 is not None:
        colors = iter(hsv(np.linspace(0, 0.9, line_num)))
        for i in range(len(legends_2)):
            _legend = legends_2[i]
            try:
                y = np.loadtxt("results/" + _legend + "/" + data_file)
            except:
                continue
            y = y[:_len]
            mean_loss = np.mean(y)
            c = next(colors)
            plt.plot(y, label=_legend + ' ' + '{:.4e}'.format(mean_loss), linewidth=1,
                     alpha=0.8, color=c, linestyle='-.', marker=markers2[(i+1) % 2],
                     markerfacecolor=c, markersize=7, markevery=7)

    sns.set_style("whitegrid", {"axes.edgecolor": "black"})

    plt.legend(loc='upper right',
               #    bbox_to_anchor=(0, 1.05),
               prop=myfont_legend,
               ncols=3,
               frameon=False)
    ax.set_ylim(10**(-8), 10**1)
    # if not is_ood:
    #     ax.set_ylim(10**(-8), 10**1)
    # else:
    #     ax.set_ylim(10**(-8), 10**4)
    ax.set_yscale('log')
    plt.grid(which='major', axis='both', linestyle='dashed')
    plt.ylabel('$(F(x_k) - F(x^*))/F(x^*)$', fontdict=myfont)
    plt.xlabel('Iteration $k$', fontdict=myfont)
    plt.xticks(fontsize=myfont['size'] // 3 * 2)
    plt.yticks(y_ticks, fontsize=myfont['size'] // 3 * 2)
    plt.tight_layout()
    plt.savefig("plots/" + name + ".pdf")


def get_directories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def get_file_with_pre(path, pre):
    return [d for d in os.listdir(path) if pre in d]
