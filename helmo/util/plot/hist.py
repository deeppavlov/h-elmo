import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import container
from matplotlib.patches import Rectangle
from plot_helpers import PatchHandler, COLORS

from learning_to_learn.useful_functions import create_path


parser = argparse.ArgumentParser()
parser.add_argument(
    'files',
    help='Path to pickle file with data.',
    nargs='+',
)
parser.add_argument(
    '--labels',
    help='Labels of data.',
    nargs='+',
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


data = {}
for label, file_name in zip(args.labels, args.files):
    data[label] = load(file_name)

for (label, x), color in zip(data.items(), COLORS):
    line = plt.hist(x, bins=1000, density=True, color=color, label=label, histtype=u'step')

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

plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
handler_map = dict(
    zip(
        handles,
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

if not args.show:
    for format in ['pdf', 'png']:
        if format == 'pdf':
            fig_path = args.output + '.pdf'
        elif format == 'png':
            fig_path = args.output + '.png'
        else:
            fig_path = None
        create_path(fig_path, file_name_is_in_path=True)
        r = plt.savefig(fig_path, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')


if args.show:
    plt.show()
