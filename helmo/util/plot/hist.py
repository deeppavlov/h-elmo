import argparse
import pickle
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import container
from matplotlib.patches import Rectangle

import helmo.util.path_help as path_help
from helmo.util.modifiers import describe_pos_tag
import plot_helpers
from plot_helpers import PatchHandler, COLORS
from learning_to_learn.useful_functions import create_path


parser = argparse.ArgumentParser()
parser.add_argument(
    'files',
    help='Path to pickle files with data.',
    nargs='+',
)
parser.add_argument(
    '--labels',
    help='Labels of data.',
    nargs='*',
)
parser.add_argument(
    "--xlabel",
    help="Label on horizontal axis. Default is 'correlation'.",
    default="correlation",
)
parser.add_argument(
    "--ylabel",
    help="Label on vertical axis. Default is 'density'",
    default='density',
)
parser.add_argument(
    "--nbins",
    "-b",
    help="Number of bins for points.",
    type=int,
    default=1000
)
parser.add_argument(
    '--output',
    '-o',
    help='Output file name without extension.'
)
parser.add_argument(
    '--lgd',
    help='Legend position. Possible options are '
         '"outside", "upper_right", "upper_left". '
         'Default is "outside".',
    default="outside",
)
parser.add_argument(
    '--show',
    '-v',
    help='If provided plots are not saved but showed.',
    action='store_true',
)
parser.add_argument(
    "--density_plot",
    "-d",
    help="If provided instead of histogram density plot is drawn. "
         "The bancdwidth is computed from `nbins` param by dividing "
         "data values diapason.",
    action="store_true",
)
parser.add_argument(
    "--no_grid",
    "-g",
    help="If provided grid is not drawn on the plot.",
    action="store_true",
)
parser.add_argument(
    "--label_modifier",
    "-m",
    help="A right hand part of an equation which modifies label.",
)
parser.add_argument(
    "--formats",
    "-f",
    help="List of formats in which image is saved. png and pdf options"
         "are available. Default is only png",
    nargs="+",
    default=["png"],
)
args = parser.parse_args()


def load(file_name):
    data = []
    with open(file_name, 'rb') as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break

    data = np.stack(data)
    return data.reshape([-1])


if args.labels is None:
    labels = [None] * len(args.files)
else:
    labels = args.labels + path_help.labels_from_paths(args.files[len(args.labels):])

if args.label_modifier is not None:
    labels = [eval(args.label_modifier.format(repr(lbl))) for lbl in labels]


data = OrderedDict()
for label, file_name in zip(labels, args.files):
    data[label] = load(file_name)

if args.density_plot:
    dt = np.concatenate([np.reshape(array, (-1,)) for array in data.values()], 0)
    bandwidth = (np.max(dt) - np.min(dt)) / args.nbins

for (label, x), color in zip(data.items(), COLORS):
    if args.density_plot:
        line = plot_helpers.density_plot(x, bandwidth, label, color)
    else:
        line = plt.hist(x, bins=args.nbins, density=True, color=color, label=label, histtype=u'step')

plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)
if not args.no_grid:
    plt.grid()

if args.labels is not None:
    # print('labels are provided')
    if args.lgd == 'outside':
        pos_dict = dict(
            bbox_to_anchor=(1.05, 1),
            loc=2,
        )
    elif args.lgd == 'upper_right':
        pos_dict = dict(
            bbox_to_anchor=(.95, .95),
            loc=1,
        )
    elif args.lgd == 'upper_left':
        pos_dict = dict(
            bbox_to_anchor=(.05, .95),
            loc=2,
        )
    ax = plt.gca()
    handles, _ = ax.get_legend_handles_labels()
    handler_map = dict(
        zip(
            sorted(handles, key=lambda x: x.get_label()),
            [
                PatchHandler(color=color) for _, color in
                zip(
                    sorted(handles, key=lambda x: x.get_label()),
                    COLORS
                )
            ]
        )
    )
    lgd = ax.legend(
        # handles,
        # labels,
        **pos_dict,
        borderaxespad=0.,
        handler_map=handler_map
    )
    bbox_extra_artists = [lgd]
else:
    bbox_extra_artists = []

if not args.show:
    for format_ in args.formats:
        if format_ == 'pdf':
            fig_path = args.output + '.pdf'
        elif format_ == 'png':
            fig_path = args.output + '.png'
        else:
            fig_path = None
        create_path(fig_path, file_name_is_in_path=True)
        r = plt.savefig(fig_path, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')


if args.show:
    plt.show()
