import os
import argparse
import pickle

from helmo.util.plot import plot_helpers

parser = argparse.ArgumentParser()

parser.add_argument(
    "plot_data",
    help="Pickle file containing plot_data dictionary.",
    type=argparse.FileType('rb'),
)
parser.add_argument(
    "--xlabel",
    "-x",
    help="Label on horizontal axis. Default is 'step'.",
    default="step",
)
parser.add_argument(
    "--ylabel",
    "-y",
    help="Label on vertical axis. Default is 'accuracy'",
    default='accuracy',
)
parser.add_argument(
    "--xscale",
    "-X",
    help="Scale of horizontal axis. Possible options are"
         " (1)linear, (2)log, (3)symlog. Default is 'linear'",
    default='linear',
)
parser.add_argument(
    "--yscale",
    "-Y",
    help="Scale of horizontal axis. Possible options are"
         " (1)linear, (2)log. Default is 'linear'",
    default='linear',
)
parser.add_argument(
    "--output",
    "-o",
    help="File name WITHOUT EXTENSION. Default is 'pickle_plot'",
    default="pickle_plot",
)
parser.add_argument(
    "--err_style",
    "-t",
    help="The way errors will be shown on plot. Possible options are"
         " (1)bar, (2)fill, (3)noerr. Default is 'bar'.",
    default='bar',
)
parser.add_argument(
    "--marker",
    "-m",
    help="Point marker as specified in https://matplotlib.org/api/markers_api.html#module-matplotlib.markers."
         " Default is ','",
    default=',',
)
parser.add_argument(
    "--no_line",
    "-n",
    help="If provided no line is drawn. Otherwise solid line style is applied.",
    action='store_true',
)
parser.add_argument(
    "--xshift",
    "-i",
    help="The number which is added to all X values. Can be used to"
         " compensate wrong step numbering. Default is 0.",
    default=0,
)
parser.add_argument(
    "--yshift",
    "-I",
    help="The number which is added to all Y values. Default is 0.",
    default=0,
)
parser.add_argument(
    "--lgd",
    "-d",
    help="Specifies legend position. Possible options are (1)outside,"
         " (2)upper_right, (3)upper_left, (4)best. Default is 'outside'.",
    default="outside",
)
parser.add_argument(
    "--only_color_as_marker_in_legend",
    "-O",
    help="If provided markers in legend will be replaced by rectangular patches"
         " of required color.",
    action="store_true",
)
parser.add_argument(
    "--lines_to_draw",
    "-l",
    help="List of labels of lines which will be plotted. By default"
         " all lines are plotted.",
    nargs='+',
    default=None,
)
parser.add_argument(
    "--xselect",
    "-S",
    help="Two numbers or one number. If `xselect` is 2 numbers,"
         " only points with X values within the ends of specified "
         "line segment are plotted. If `xselect` one number only points with X more"
         " than `xselect` are plotted.",
    nargs='+',
    type=float,
)
parser.add_argument(
    "--save_formats",
    "-s",
    nargs='+',
    help="List of formats in which plot will be saved. Possible options are"
         " (1)png, (2)pdf. At list one format has to specified.",
    default=['pdf', 'png'],
)
parser.add_argument(
    "--exec_code",
    "-e",
    help="Code that needs to be executed for correct plot_data loading. "
         "It is provided as python file name."
)
parser.add_argument(
    "--dpi",
    "-r",
    help="Saved image resolution.",
    type=int,
    default=300,
)
parser.add_argument(
    "--size_factor",
    "-f",
    help="The factor by which figure size will be increased. This allows "
         "to see more details on the plot.",
    type=float,
    default=1.,
)
parser.add_argument(
    "--grid",
    "-g",
    help="If provided grid is drawn.",
    action="store_true",
)
parser.add_argument(
    "--which_grid",
    "-w",
    help="Which grid is shown if --grid option is set. "
         "Works as `which` parameter in "
         "`matplotlib.pyplot.grid()` function. Possible"
         " options are (1)major, (2)minor, (3)both. Default is"
         " major.",
    default='major',
)
parser.add_argument(
    "--only_show",
    "-v",
    help="If specified the plot is not saved but showed.",
    action="store_true",
)
parser.add_argument(
    "--additional_artists",
    "-a",
    help="Path to pickle file with specs of additional objects on plot."
         " File has to contain a dictionary with lists of specs of"
         " various artists. Possible keys in the dictionary: (1)axvspan "
         "is for drawing  vertical spans with matplotlib.pyplot.axvspan().",
)
parser.add_argument(
    "--linewidth",
    help="Line width in points. Default is 1.0",
    type=float,
    default=1.0,
)
args = parser.parse_args()

if args.xselect is not None:
    args.xselect = args.xselect[:2]
    if len(args.xselect) == 1:
        args.xselect.append(float('inf'))

if args.exec_code is not None:
    import importlib.util
    module_name = os.path.split(os.path.splitext(args.exec_code)[0])[0]
    spec = importlib.util.spec_from_file_location(module_name, args.exec_code)
    exec_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exec_module)
    for attr_name in dir(exec_module):
        if attr_name[0] != '_':
            exec(attr_name + ' = exec_module.' + attr_name)

plot_data = pickle.load(args.plot_data)

plot_data = plot_helpers.PlotData.old_format_to_new_format(plot_data)

if args.additional_artists is not None:
    with open(args.additional_artists, 'rb') as f:
        additional_artists = pickle.load(f)
else:
    additional_artists = None

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
        linewidth=args.linewidth,
    ),
    shifts=[args.xshift, args.yshift],
    legend_pos=args.lgd,
    only_color_as_marker_in_legend=args.only_color_as_marker_in_legend,
    labels_of_drawn_lines=args.lines_to_draw,
    save=not args.only_show,
    show=args.only_show,
    select=args.xselect,
    dpi=args.dpi,
    size_factor=args.size_factor,
    grid=args.grid,
    which_grid=args.which_grid,
    formats=args.save_formats,
    additional_artists=additional_artists,
)
