import argparse
import copy
import string
from collections import OrderedDict

import matplotlib.pyplot as plt


COLORS = [
    'red', 'green', 'blue', 'black', 'cyan', 'magenta', 'brown',
    'darkviolet', 'pink', 'yellow', 'gray', 'orange', 'olive',
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "table_file_name",
        help="Path to file with table",
    )
    parser.add_argument(
        "--tag_file_name",
        "-l",
        help="Path to file with tags in which"
             " tags are listed each in a separate line. "
             "Used for labelling of horizontal axis.",
    )
    parser.add_argument(
        "--tag_type",
        "-t",
        help="For standard tagging such as tagging by "
             "lowercase letter. Used for labelling "
             "of horizontal axis."
    )
    parser.add_argument(
        "--upper_tags_file_name",
        "-u",
        help="Path to file with upper tags. Upper tags are "
             "put on additional X axis on the top of the plot."
    )
    parser.add_argument(
        "--xlabel",
        "-x",
        help="Name of the horizontal axis. Default is 'part of speech'",
        default="part of speech"
    )
    parser.add_argument(
        "--ylabel",
        "-y",
        help="Name of the vertical axis. Default is 'correlation'",
        default="correlation",
    )
    parser.add_argument(
        "--layer_names",
        "-n",
        help="Layer names shown on plot. Default is 'layer #1', 'layer #2' "
             "and so on.",
        nargs="+",
    )
    parser.add_argument(
        "--markers",
        "-m",
        help="Markers of points on the plot. You may provide indifidual"
             " markers for every layer or specify one marker for all. "
             "Default is circle.",
        nargs='+',
        default='o',
    )
    parser.add_argument(
        "--colors",
        "-c",
        help="Colors of points for different layers. Default color order "
             "is {}".format(COLORS),
        nargs='+',
    )
    parser.add_argument(
        "--only_show",
        "-v",
        help="Plot is not saved but only shown.",
        action="store_true",
    )
    parser.add_argument(
        "--save_path",
        "-s",
        help="Path to file where plot will be saved. Default is 'tags.png'",
        default="tags.png",
    )
    parser.add_argument(
        "--figure_shape",
        "-f",
        help="Shape of drawn figure in inches. Default is [12, 4].",
        nargs='+',
        type=int,
        default=[12, 4],
    )
    parser.add_argument(
        "--dpi",
        "-r",
        help="Image resolution in dpi. Default is 900.",
        type=int,
        default=900,
    )
    args = parser.parse_args()
    return args


def parse_pos_corr_table(table_file_name, layer_names):
    layer_names = copy.deepcopy(layer_names)
    with open(table_file_name) as f:
        table_text = f.read()
    text_by_tags = table_text.split('\n')
    text_by_tags = [line for line in text_by_tags if line]
    n = len(text_by_tags[0].split(';'))
    if layer_names is None:
        layer_names = ['layer #{}'.format(i) for i in range(n)]
    data = OrderedDict(zip(layer_names, [[[], [], []] for _ in layer_names]))
    for i, line in enumerate(text_by_tags):
        values_and_errors = line.split(';')
        for layer_name, vne in zip(layer_names, values_and_errors):
            v, e = vne.split(' ± ')
            v = float(v)
            e = float(e)
            data[layer_name][0].append(i)
            data[layer_name][1].append(v)
            data[layer_name][2].append(e)
    return data


def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def mark_x_axis(tags):
    n = len(tags)
    plt.xticks(range(n))
    plt.gca().set_xticklabels(tags)


def read_tags(tag_file_name):
    with open(tag_file_name) as f:
        tags = f.readlines()
    return [t.strip() for t in tags]


def create_tags(tag_type):
    if tag_type == 'lowercase':
        tags = list(string.ascii_lowercase)
    else:
        raise ValueError("Unsupported tag type")
    return tags


def add_upper_x_axis(upper_tags):
    xlim = plt.xlim()
    secax = plt.twiny()
    secax.set_xlim(*xlim)
    secax.set_xticks(range(len(upper_tags)))
    secax.set_xticklabels(upper_tags, rotation='vertical')


def tag_plot(data, tags, colors, markers, xlabel, ylabel, upper_tags=None):
    _, ax = plt.subplots()
    for (label, dt), color, mk in zip(data.items(), colors, markers):
        ax.errorbar(dt[0], dt[1], yerr=dt[2], color=color, marker=mk, label=label, linestyle=' ')
    mark_x_axis(tags)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if upper_tags is not None:
        add_upper_x_axis(upper_tags)
    ax.grid()
    ax.legend(loc='best')


def get_num_tags(data):
    return len(list(data.values())[0][0])


def prepare_markers(markers, num_layers):
    if len(markers) == 1:
        markers = [markers[0]]*num_layers
    elif len(markers) == num_layers:
        pass
    else:
        ValueError('Number of markers is not 1 and does not match number of layers.')
    return markers


def main():
    args = get_args()
    data = parse_pos_corr_table(
        args.table_file_name,
        args.layer_names,
    )
    num_tags = get_num_tags(data)
    num_layers = len(data)
    if args.tag_file_name is None:
        tags = create_tags(args.tag_type)
    else:
        tags = read_tags(args.tag_file_name)
    if args.upper_tags_file_name is None:
        upper_tags = None
    else:
        upper_tags = read_tags(args.upper_tags_file_name)[:num_tags]
    markers = prepare_markers(args.markers, num_layers)
    tag_plot(
        data,
        tags[:num_tags],
        COLORS[:num_layers],
        markers,
        args.xlabel,
        args.ylabel,
        upper_tags=upper_tags,
    )
    set_size(*args.figure_shape)
    plt.tight_layout()
    if args.only_show:
        plt.show()
    else:
        plt.savefig(args.save_path, dpi=args.dpi)
    return markers


if __name__ == '__main__':
    main()
