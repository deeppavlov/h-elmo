import argparse
import pickle

from helmo.util import interpreter
interpreter.extend_python_path_for_project()

from helmo.util.plot import plot_helpers

parser = argparse.ArgumentParser()

parser.add_argument(
    "plot_data",
    help="Pickle file containing plot_data dictionary.",
    type=argparse.FileType('rb'),
)
parser.add_argument(
    "--xlabel",
    help="Label on horizontal axis. Default is 'step'.",
    default="step",
)
parser.add_argument(
    "--ylabel",
    help="Label on vertical axis. Default is 'accuracy'",
    default='accuracy',
)
parser.add_argument(
    "--xscale",
    help="Scale of horizontal axis. Possible options are"
         " (1)linear, (2)log, (3)symlog. Default is 'linear'",
    default='linear',
)
parser.add_argument(
    "--yscale",
    help="Scale of horizontal axis. Possible options are"
         " (1)linear, (2)log. Default is 'linear'",
    default='linear',
)
parser.add_argument(
    "--output",
    help="File name WITHOUT EXTENSION. Default is 'pickle_plot'",
    default="pickle_plot",
)
parser.add_argument(
    "--err_style",
    help="The way errors will be shown on plot. Possible options are"
         " (1)bar, (2)fill, (3)noerr. Default is 'bar'.",
    default='bar',
)
parser.add_argument(
    "--marker",
    help="Point marker as specified in https://matplotlib.org/api/markers_api.html#module-matplotlib.markers."
         " Default is ','",
    default=',',
)
parser.add_argument(
    "--no_line",
    help="If provided no line is drawn. Otherwise solid line style is applied.",
    action='store_true',
)
parser.add_argument(
    "--xshift",
    help="The number which is added to all X values. Can be used to"
         " compensate wrong step numbering. Default is 0.",
    default=0,
)
parser.add_argument(
    "--yshift",
    help="The number which is added to all Y values. Default is 0.",
    default=0,
)
parser.add_argument(
    "--lgd",
    help="Specifies legend position. Possible options are (1)outside,"
         " (2)upper_right, (3)upper_left. Default is 'outside'.",
    default="outside",
)
parser.add_argument(
    "--lines_to_draw",
    help="List of labels of lines which will be plotted. By default"
         " all lines are plotted.",
    nargs='+',
    default=None,
)
parser.add_argument(
    "--save_formats",
    nargs='+',
    help="List of formats in which plot will be saved. Possible options are"
         " (1)png, (2)pdf. At list one format has to specified. if you wish to"
         " only show plot use -s option.",
    default=['pdf', 'png'],
)
parser.add_argument(
    "--only_show",
    "-v",
    help="If specified the plot is not saved but showed.",
    action="store_true",
)
args = parser.parse_args()

plot_data = pickle.load(args.plot_data)

plot_helpers.plot_outer_legend(
    plot_data,
    None,
    args.xlabel,
    args.ylabel,
    args.xscale,
    args.yscale,
    args.output,
    dict(
        no_line=args.no_line,
        error=args.err_style,
        marker=args.marker,
    ),
    shifts=[args.xshift, args.yshift],
    legend_pos=args.lgd,
    labels_of_drawn_lines=args.lines_to_draw,
    save=not args.only_show,
    show=args.only_show,
)
