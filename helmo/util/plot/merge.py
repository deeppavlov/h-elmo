import argparse
import pickle

import helmo.util.plot.plot_helpers as plot_helpers

parser = argparse.ArgumentParser()
parser.add_argument(
    'xsrc',
    help="File with X values. It can be .txt or .pickle. If it is .txt"
         " colx have to be provided",
)
parser.add_argument(
    'ysrc',
    help="File with Y values. It can .txt or .pickle. If it is .txt"
         " coly have to be provided",
)
parser.add_argument(
    '--colx',
    help="Number of a column in srcx if it is text file. Numbering is "
         "starting with 0",
    type=int,
)
parser.add_argument(
    '--coly',
    help="Number of a column in srcy if it is text file.",
    type=int,
)
parser.add_argument(
    '--xerrsrc',
    help="File with X error values. Have to be provided if xsrc is pickle",
)
parser.add_argument(
    '--yerrsrc',
    help="File with X error values. Have to be provided if ysrc is pickle",
)
parser.add_argument(
    '--xerrcol',
    help="Number of X error column in xsrc if xsrc is txt file",
    type=int,
)
parser.add_argument(
    '--yerrcol',
    help="Number of X error column in ysrc if ysrc is txt file",
    type=int,
)
parser.add_argument(
    '--label',
    help="label of the line",
)
parser.add_argument(
    '--output',
    help="Path to file where results will be stored. Points will have"
         " format 'x mean_y stddev_y, stderr_of_mean_y'. Default is"
         " merge.pickle",
    type=argparse.FileType('wb'),
    default='merge.pickle'
)
parser.add_argument(
    '--reverse',
    help="If provided points are sorted in reverse order.",
    action='store_true',
)

args = parser.parse_args()


def fill_values_and_err(src, col, errsrc, errcol):
    if src[-4:] == '.txt':
        v = []
        err = []
        with open(src) as f:
            for l in f:
                values = [float(v) for v in l.split()]
                v.append(values[col])
                if errcol is not None:
                    err.append(values[errcol])
                else:
                    err.append(0.)
    else:
        with open(src, 'rb') as f:
            v = pickle.load(f)
        if errsrc is None:
            err = [0.]*len(v)
        else:
            with open(errsrc, 'rb') as f:
                err = pickle.load(f)
    return v, err


x, x_err = fill_values_and_err(args.xsrc, args.colx, args.xerrsrc, args.xerrcol)
y, y_err = fill_values_and_err(args.ysrc, args.coly, args.yerrsrc, args.yerrcol)

# for xx, yy, xx_err, yy_err in sorted(zip(x, y, x_err, y_err), key=lambda x: x[0], reverse=args.reverse):
#     args.output.write('{} {} {} {}\n'.format(xx, yy, xx_err, yy_err))

x, y, x_err, y_err = zip(*sorted(zip(x, y, x_err, y_err), key=lambda x: x[0], reverse=args.reverse))
# print("(merge.py)x_err:", x_err)
# print("(merge.py)y_err:", y_err)
pd = plot_helpers.PlotData(
    [(args.label, dict(x=x, y=y, x_err=x_err, y_err=y_err))]
)
# print('x, xerr')
# for xx, xxerr1, xxerr2 in zip(pd['dropout 0']['x'], pd['dropout 0']['x_err'][0], pd['dropout 0']['x_err'][1]):
#     print(xx, xxerr1, xxerr2)
# print('\ny, yerr')
# for yy, yyerr1, yyerr2 in zip(pd['dropout 0']['y'], pd['dropout 0']['y_err'][0], pd['dropout 0']['y_err'][1]):
#     print(yy, yyerr1, yyerr2)
# print("(merge.py)pd['dropout 0']['y']:", pd['dropout 0']['y'])
pickle.dump(
    pd,
    args.output,
)

