import os
import random
import warnings
import copy
import itertools
from collections import UserDict
from typing import List

import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt, rc
from matplotlib.legend_handler import HandlerLine2D
from matplotlib import container
import matplotlib.patches as mpatches

import helmo.util.python as python
from helmo.util.algo import SortedDict, sorting_key_float
from learning_to_learn.useful_functions import synchronous_sort, create_path, get_pupil_evaluation_results, \
    BadFormattingError, all_combs, get_optimizer_evaluation_results, select_for_plot, convert, retrieve_lines, \
    add_index_to_filename_if_needed, nested2string, isnumber, add_scalar_iterable


# from pathlib import Path  # if you haven't already done so
# file = Path(__file__).resolve()
# parent, root = file.parent, file.parents[2]
# sys.path.append(str(root))
# try:
#     sys.path.remove(str(parent))
# except ValueError:  # Already removed
#     pass


COLORS = [
    'r', 'g', 'b', 'k', 'c', 'magenta', 'brown',
    'darkviolet', 'pink', 'yellow', 'gray', 'orange', 'olive',
]
DPI = 900
FORMATS = ['pdf', 'png']
AVERAGING_NUMBER = 3

FONT = {'family': 'Verdana',
        'weight': 'normal'}
rc('font', **FONT)


class Line(UserDict):
    value_keys = ['x', 'y']
    error_keys = ['x_err', 'y_err']
    value_error_pairs = dict(zip(value_keys, error_keys))
    axes = ['x', 'y']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # print("(Line.__init__)args:", args)
        # print("(Line.__init__)self.keys():", list(self.keys()))
        if not isinstance(args[0], Line):
            for key in self.keys():
                # print("(Line.__init__)key:", key)
                if key in self.value_keys:
                    self.expand_values(key)
                    if len(self['x']) != len(self['y']):
                        raise ValueError("numbers of x and y values are not equal")
                    if len(self['x']) == 0:
                        raise ValueError("no points were provided")
                    if self.value_error_pairs[key] in self:
                        self.expand_errors(key)
        # print("(Line.__init__)self['y']:", self['y'])
        # print("(Line.__init__)self['y_err']:", self['y_err'])

    @staticmethod
    def old_format_to_new_format(line_data):
        if not isinstance(line_data, (list, tuple)):
            return line_data
        return Line(
            x=line_data[0],
            y=line_data[1],
            y_err=line_data[2] if len(line_data) > 2 else None,
        )
    
    def expand_errors(self, value_key):
        error_key = self.value_error_pairs[value_key]
        # print("(Line.expand_errors)error_key:", error_key)
        # print("(Line.expand_errors)self[error_key]:", self[error_key])
        err_msg = (
            "wrong '{}' error format. Error has to be either "
            "list of numbers or iterable of 2 iterables of equal length.".format(error_key)
        )
        if not python.is_iterable(self[error_key]):
            self[error_key] = [self[error_key]] * len(self[value_key])
        else:
            self[error_key] = list(self[error_key])
        if all([python.is_iterable(elem) for elem in self[error_key]]):
            errors = self[error_key]
            if len(self[error_key]) == 2:
                if len(errors[0]) != len(errors[1]):
                    raise ValueError(
                        err_msg + "Lengths of iterables are not equal.\nlen(a) == {}\nlen(b) == {}".format(
                            len(errors[0]), len(errors[1])
                        )
                    )
                self[error_key] = [list(errors[0]), list(errors[1])]
            else:
                raise ValueError(err_msg + " More than 2 iterables are provided")
        else:
            if any([python.is_iterable(elem) for elem in self[error_key]]):
                raise ValueError(err_msg + " Some elements of error are iterables and some are not.")
            self[error_key] = [list(self[error_key]), list(self[error_key])]
        
    def expand_values(self, value_key):
        if python.is_iterable(self[value_key]):
            self[value_key] = list(self[value_key])
        else:
            self[value_key] = [self[value_key]]

    def shift_line(self, axis, shift):
        if axis not in self.axes:
            raise ValueError("not supported axis {}".format(axis))
        self[axis] = python.add_scalar_to_iterable(self[axis], shift)
        error_key = Line.value_error_pairs[axis]
        if error_key in self:
            # print("(Line.shift_line)self[error_key]:", self[error_key])
            lower, upper = self[error_key]
            lower = python.add_scalar_to_iterable(lower, shift)
            upper = python.add_scalar_to_iterable(upper, shift)
            self[error_key] = [lower, upper]

    def get_bounds(self):
        if 'x_err' in self:
            raise NotImplementedError("bounds computation is not implemented when 'x_err' is not zero")
        else:
            if 'y_err' in self:
                lower_err, upper_err = self['y_err']
                lower = python.subtract_iterable(self['y'], lower_err)
                upper = python.add_iterable(self['y'], upper_err)
                return (self['x'].copy(), lower), (self['x'].copy(), upper)
            else:
                return (self['x'].copy(), self['y'].copy()), (self['x'].copy(), self['y'].copy())

    def get_all_values(self, spec):
        if spec in self.value_keys:
            return self[spec].copy()
        elif spec in self.error_keys:
            return list(itertools.chain(*self[spec]))
        else:
            raise ValueError("spec {} is not supported".format(spec))

    def _create_ind_gen(self, select):
        x_list = self['x']

        def index_gen():
            for i, x in enumerate(x_list):
                if select[0] <= x <= select[1]:
                    yield i
        return index_gen
        
    def filter_points(self, select):
        index_gen = self._create_ind_gen(select)
        for key in self.value_keys:
            self[key] = [self[key][i] for i in index_gen()]
        for key in self.error_keys:
            if key in self:
                self[key] = [
                    [self[key][0][i] for i in index_gen()],
                    [self[key][1][i] for i in index_gen()]
                ]

    def __repr__(self):
        elements = ', '.join(['({}, {})'.format(repr(k), repr(v)) for k, v in self.items()])
        return '{}([{}])'.format(self.__class__.__name__, elements)

    def __str__(self):
        return repr(self)


class PlotData(SortedDict):
    value_keys = Line.value_keys
    error_keys = Line.error_keys

    @staticmethod
    def old_format_to_new_format(plot_data):
        if isinstance(plot_data, PlotData):
            return plot_data
        new_init = []
        for lbl, line_data in plot_data.items():
            new_init.append(
                (
                    lbl,
                    Line.old_format_to_new_format(line_data)
                )
            )
        return PlotData(new_init)

    def _transform_key_to_str(self, key):
        if not isinstance(key, str):
            new_key = str(key)
            if new_key in self:
                warnings.warn(
                    "replacing value of element `('{}', VALUE)` with"
                    " value of element `({}, VALUE)`. Conflict between"
                    " line labels occured during transforming labels to type"
                    " str. The element `({}, VALUE)` is going to be "
                    "removed".format(new_key, key, key)
                )
            return new_key

    def get_labels(self):
        return list(self.keys())

    def get_lines(self):
        return list(self.values())

    def __setitem__(self, key, value):
        # print("(PlotData.__setitem__)value:", value)
        if not isinstance(key, str):
            warnings.warn(
                "only keys of type `str` are allowed."
                " Element with key '{}' will be set.".format(str(key))
            )
            key = self._transform_key_to_str(key)
        super().__setitem__(str(key), Line(value))

    def get_spec(self, spec, labels=None):
        if labels is None:
            labels = self.get_labels()
        res = []
        for lbl in labels:
            try:
                line_data = self[lbl]
            except KeyError:
                raise KeyError("label '{}' is not in plot data".format(lbl))
            try:
                res.append(line_data[spec])
            except KeyError:
                raise KeyError(
                    "spec '{}' is not in line data"
                    " with label '{}'".format(spec, lbl)
                )
        return res

    def get_all_values(self, spec):
        res = []
        for line_data in self.values():
            if spec in line_data:
                res += line_data.get_all_values(spec)
        return res

    def __copy__(self):
        pd = PlotData()
        pd.__dict__.update(self.__dict__)
        return pd

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        pd = PlotData()
        memo[id(pd)] = pd
        for k, v in self.__dict__:
            setattr(pd, k, copy.deepcopy(v, memo))
        return pd

    def all_labels_are_numbers(self):
        return all([isnumber(k) for k in self])

    def set_float_sorting_key(self):
        self.set_sorting_key(sorting_key_float)

    def labels_are_provided(self):
        there_is_labels = False
        for label in self.keys():
            if len(label) > 0:
                there_is_labels = there_is_labels or True
        return there_is_labels

    def _prepare_shifts(self, shifts):
        if isinstance(shifts, (float, int)):
            shifts = dict(zip(self.keys(), [shifts] * len(self)))
        elif isinstance(shifts, (list, tuple)):
            shifts = dict(zip(self.keys(), shifts))
        elif isinstance(shifts, (dict, SortedDict)):
            pass
        else:
            raise TypeError("unsupported shift type")
        return shifts

    def shift_lines(self, axis, shifts):
        shifts = self._prepare_shifts(shifts)
        for label, sh in shifts.items():
            self[label].shift_line(axis, sh)

    def get_bounds(self, label_or_labels):
        if label_or_labels is None:
            label_or_labels = self.keys()
        if not isinstance(label_or_labels, str):
            return self[label_or_labels].get_bounds()
        else:
            return [self[label].get_bounds() for label in label_or_labels]
        
    def filter_points(self, select):
        for line in self.values():
            line.filter_points(select)


def get_parameter_name(plot_parameter_names, key):
    try:
        v = plot_parameter_names[key]
    except KeyError:
        warnings.warn("no '%s' entry parameter names file" % key)
        v = key
    return v


def fixed_hps_from_str(string):
    if len(string) > 0:
        tmp = string[1:-1]
        hps = [convert(x, "float") for x in tmp]
        return tuple(hps)
    else:
        return ()


def parse_metric_scales_str(string):
    metric_scales = dict()
    if string is not None:
        for one_metric_scale in string.split(','):
            [metric, scale] = one_metric_scale.split(':')
            metric_scales[metric] = scale
    return metric_scales


def get_linthresh(values: List) -> float:
    """
    Return the smallest absolute value of X or Y in list `values`.
    Used to compute `linthresh` param of `matplotlib.pyplot.scale()`
    when 'symlog' scale is used.

    Args:
        values: list of numbers.

    Returns:
        `linthresh` - the smallest absolute value of X.
    """
    thresh = float('inf')
    for v in values:
        av = abs(v)
        if 0 < av < thresh:
            thresh = av
    return thresh


class PatchHandler:
    def __init__(self, color=None):
        self._color = color

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle(
            [x0, y0], width, height,
            facecolor=orig_handle.get_color() if self._color is None else self._color,
            transform=handlebox.get_transform()
        )
        handlebox.add_artist(patch)
        return patch


def add_legend(artists, position, only_color_as_marker_in_legend):
    if position == 'outside':
        pos_dict = dict(
            bbox_to_anchor=(1.05, 1),
            loc=2,
        )
    elif position == 'upper_right':
        pos_dict = dict(
            bbox_to_anchor=(.95, .95),
            loc=1,
        )
    elif position == 'upper_left':
        pos_dict = dict(
            bbox_to_anchor=(.05, .95),
            loc=2,
        )
    if only_color_as_marker_in_legend:
        handler_map = dict(list(zip(artists, [PatchHandler() for _ in range(len(artists))])))
    else:
        handler_map = dict(list(zip(artists, [HandlerLine2D(numpoints=1) for _ in range(len(artists))])))
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    lgd = ax.legend(
        handles,
        labels,
        **pos_dict,
        borderaxespad=0.,
        handler_map=handler_map,
    )
    return lgd


def plot_outer_legend(
        plot_data,
        description,
        xlabel,
        ylabel,
        xscale,
        yscale,
        file_name_without_ext,
        style,
        shifts=None,
        legend_pos='outside',
        only_color_as_marker_in_legend=False,
        labels_of_drawn_lines=None,
        formats=None,
        save=True,
        show=False,
        axes_to_invert=(),
        select=None,
        dpi=300,
        size_factor=1,
        grid=False,
        which_grid='major',
):
    if shifts is None:
        shifts = [0, 0]
    if style['no_line']:
        linestyle = 'None'
    else:
        linestyle = 'solid'
    if select is not None:
        plot_data.filter_points(select)

    plot_data.shift_lines('x', shifts[0])
    plot_data.shift_lines('y', shifts[1])
    if labels_of_drawn_lines is not None:
        plot_data = python.filter_dict_by_keys(plot_data, labels_of_drawn_lines)
    if formats is None:
        formats = FORMATS
    rc('font', **FONT)
    plt.clf()
    plt.subplot(111)
    if plot_data.all_labels_are_numbers():
        plot_data.set_float_sorting_key()
    lines = list()
    for idx, (label, line_data) in enumerate(plot_data.items()):
        if label is None or label == 'None':
            label = ''
        if idx > len(COLORS) - 1:
            color = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        else:
            color = COLORS[idx]

        if style['error'] == 'fill':
            bounds = line_data.get_bounds()
            x_err, y_err = None, None
            plt.fill_between(
                bounds[0][0],
                bounds[0][1],
                bounds[1][1],
                alpha=.4,
                color=color,
            )
        elif style['error'] == 'bar':
            x_err, y_err = line_data.get('x_err'), line_data.get('y_err')
        else:
            x_err, y_err = None, None
        # print("(plot_helpers.plot_outer_legend)line_data['y_err']:", line_data['y_err'])
        # print("(plot_helpers.plot_outer_legend)line_data['x'][:10]:", line_data['x'][:10])
        # print("(plot_helpers.plot_outer_legend)line_data['y'][:10]:", line_data['y'][:10])
        lines.append(
            plt.errorbar(
                line_data['x'],
                line_data['y'],
                yerr=y_err,
                xerr=x_err,
                marker=style['marker'],
                color=color,
                label=label,
                ls=linestyle,
            )[0]
        )

    if 'x' in axes_to_invert:
        plt.gca().invert_xaxis()
    if 'y' in axes_to_invert:
        plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    scale_kwargs = dict()
    if xscale == 'symlog':
        linthreshx = get_linthresh(
            plot_data.get_all_values('x') + plot_data.get_all_values('x_err'),
        )
        scale_kwargs['linthreshx'] = linthreshx
    plt.xscale(xscale, **scale_kwargs)
    plt.yscale(yscale)

    plt.grid(b=grid, which=which_grid)

    if plot_data.labels_are_provided():
        bbox_extra_artists = [add_legend(lines, legend_pos, only_color_as_marker_in_legend)]
    else:
        bbox_extra_artists = ()
    fig = plt.gcf()
    size = fig.get_size_inches()
    fig.set_size_inches(size[0]*size_factor, size[1]*size_factor)
    if save:
        for format in formats:
            if format == 'pdf':
                fig_path = file_name_without_ext + '.pdf'
            elif format == 'png':
                fig_path = file_name_without_ext + '.png'
            else:
                fig_path = None
            create_path(fig_path, file_name_is_in_path=True)
            r = plt.savefig(
                fig_path,
                bbox_extra_artists=bbox_extra_artists,
                bbox_inches='tight',
                dpi=dpi,
            )
    if show:
        plt.show()
    if description is not None:
        description_file = os.path.join(file_name_without_ext + '.txt')
        with open(description_file, 'w') as f:
            f.write(description)


def get_parameter_names(conf_file):
    old_dir = os.getcwd()
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    with open(conf_file, 'r') as f:
        t = f.read()
    os.chdir(old_dir)
    lines = t.split('\n')
    idx = 0
    num_lines = len(lines)
    plot_parameter_names = dict()
    while idx < num_lines and len(lines[idx]) > 0:
        spl = lines[idx].split()
        inner_name, plot_name = spl[0], spl[1:]
        plot_name = ' '.join(plot_name)
        plot_parameter_names[inner_name] = plot_name
        idx += 1
    # print("(plot_helpers.get_parameter_names)plot_parameter_names:", plot_parameter_names)
    return plot_parameter_names


def create_plot_hp_layout(plot_dir, hp_plot_order, changing_hp):
    file_with_hp_layout_description = os.path.join(plot_dir, 'plot_hp_layout.txt')
    num_of_hps = len(hp_plot_order)
    if num_of_hps > 2:
        tmpl = '%s ' * (num_of_hps - 3) + '%s'
    else:
        tmpl = ''
    if num_of_hps > 1:
        line_hp_name = hp_plot_order[-2]
    else:
        line_hp_name = ''
    with open(file_with_hp_layout_description, 'w') as f:
        f.write('fixed hyper parameters: ' + tmpl % tuple(hp_plot_order[:-2]) + '\n')

        f.write('line hyper parameter: ' + line_hp_name + '\n')
        f.write('changing hyper parameter: ' + changing_hp)
    return tmpl


def get_y_specs(res_type, plot_parameter_names, metric_scales):
    ylabel = get_parameter_name(plot_parameter_names, res_type)
    if res_type in metric_scales:
        yscale = metric_scales[res_type]
    else:
        yscale = 'linear'
    return ylabel, yscale


def launch_plotting(data, line_label_format, fixed_hp_tmpl, path, xlabel, ylabel, xscale, yscale, style, select):
    if select is not None:
        data = select_for_plot(data, select)
    on_descriptions = dict()
    for fixed_hps_tuple, plot_data in data.items():
        plot_data_on_labels = dict()
        for line_hp_value, line_data in plot_data.items():
            label = line_label_format.format(line_hp_value)
            if label in plot_data_on_labels:
                warnings.warn(
                    "specified formatting does not allow to distinguish '%s' in legend\n"
                    "fixed_hps_tuple: %s\n"
                    "falling to string formatting" % (line_hp_value, fixed_hps_tuple)
                )
                label = '%s' % line_hp_value
                if label in plot_data_on_labels:
                    raise BadFormattingError(
                        line_label_format,
                        line_hp_value,
                        "Specified formatting does not allow to distinguish '%s' in legend\n"
                        "fixed_hps_tuple: %s\n"
                        "String formatting failed to fix the problem" % (line_hp_value, fixed_hps_tuple)
                    )
            plot_data_on_labels[line_label_format.format(line_hp_value)] = line_data
        on_descriptions[fixed_hp_tmpl % fixed_hps_tuple] = plot_data_on_labels
        # print("(plot_helpers.plot_hp_search)plot_data:", plot_data)

    counter = 0
    for description, plot_data in on_descriptions.items():
        file_name_without_ext = os.path.join(path, str(counter))
        plot_outer_legend(
            plot_data, description, xlabel, ylabel, xscale, yscale, file_name_without_ext, style
        )
        counter += 1


def plot_hp_search_optimizer(
        eval_dir,
        plot_dir,
        hp_plot_order,
        plot_parameter_names,
        metric_scales,
        xscale,
        style,
        line_label_format,
        select,
):
    changing_hp = hp_plot_order[-1]
    for_plotting = get_optimizer_evaluation_results(eval_dir, hp_plot_order, AVERAGING_NUMBER)
    pupil_names = sorted(list(for_plotting.keys()))
    result_types = sorted(list(for_plotting[pupil_names[0]].keys()))
    regimes = sorted(list(for_plotting[pupil_names[0]][result_types[0]].keys()))
    fixed_hp_tmpl = create_plot_hp_layout(plot_dir, hp_plot_order, changing_hp)
    # print("(plot_hp_search)plot_parameter_names:", plot_parameter_names)
    xlabel = get_parameter_name(plot_parameter_names, changing_hp)

    for pupil_name in pupil_names:
        for res_type in result_types:
            ylabel, yscale = get_y_specs(res_type, plot_parameter_names, metric_scales)
            for regime in sorted(regimes):
                path = os.path.join(plot_dir, pupil_name, res_type, regime)
                create_path(path)
                data = for_plotting[pupil_name][res_type][regime]
                launch_plotting(
                    data, line_label_format, fixed_hp_tmpl, path, xlabel, ylabel, xscale, yscale, style, select
                )


def plot_hp_search_pupil(
        eval_dir,
        plot_dir,
        hp_plot_order,
        plot_parameter_names,
        metric_scales,
        xscale,
        style,
        line_label_format,
        select,
):
    changing_hp = hp_plot_order[-1]
    for_plotting = get_pupil_evaluation_results(eval_dir, hp_plot_order)
    dataset_names = sorted(list(for_plotting.keys()))
    result_types = sorted(list(for_plotting[dataset_names[0]].keys()))
    fixed_hp_tmpl = create_plot_hp_layout(plot_dir, hp_plot_order, changing_hp)
    xlabel = get_parameter_name(plot_parameter_names, changing_hp)
    for dataset_name in dataset_names:
        for res_type in result_types:
            ylabel, yscale = get_y_specs(res_type, plot_parameter_names, metric_scales)
            path = os.path.join(plot_dir, dataset_name, res_type)
            create_path(path)
            data = for_plotting[dataset_name][res_type]
            launch_plotting(
                data, line_label_format, fixed_hp_tmpl, path, xlabel, ylabel, xscale, yscale, style, select
            )


def plot_lines_from_diff_hp_searches(
        line_retrieve_inf,
        plot_dir,
        changing_hp,
        plot_parameter_names,
        metric_scales,
        xscale,
        style,
        x_select,
        model,
):
    # print(line_retrieve_inf)
    lines = retrieve_lines(line_retrieve_inf, x_select, model, AVERAGING_NUMBER)
    xlabel = get_parameter_name(plot_parameter_names, changing_hp)
    create_path(plot_dir)
    plot_description_file = os.path.join(plot_dir, 'description.txt')
    with open(plot_description_file, 'w') as f:
        f.write(nested2string(line_retrieve_inf))
    for res_type, plot_data in lines.items():
        ylabel, yscale = get_y_specs(res_type, plot_parameter_names, metric_scales)
        file_name_without_ext = add_index_to_filename_if_needed(os.path.join(plot_dir, res_type))
        plot_outer_legend(
            plot_data, None, xlabel, ylabel, xscale, yscale, file_name_without_ext, style
        )


def density_plot(data, bandwidth, label, color, range_=None):
    min_ = min(data) if range_ is None else range_[0]
    max_ = max(data) if range_ is None else range_[1]
    xs = np.linspace(min_, max_, int((max_ - min_) / bandwidth))

    density = gaussian_kde(data)
    density.covariance_factor = lambda: bandwidth
    return plt.plot(xs, density(np.reshape(xs, (1, -1))), label=label, color=color)
