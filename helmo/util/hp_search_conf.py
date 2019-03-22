import os
import argparse

import helmo.util.path_help

from learning_to_learn.useful_functions import create_path, parse_path_comb, get_points_from_range, get_tmpl


parser = argparse.ArgumentParser()

parser.add_argument(
    "confs",
    help="Paths to created configs used for hp search. \nTo process several configs"
         " use following format '<path1>,<path2>,...<pathi>:<pathi+1>,..:..'.\nAll possible combinations of sets"
         " separated by colons (in specified order) will be processed. Combinations are formed in the following way: "
         "from each set one name is chosen\nYou have to provide paths relative to "
         "script. Edge characters of <path> can't be '/'"
)
parser.add_argument(
    '-hp',
    "--hyper_parameters",
    help="Hyper parameters in conf. You have to specify hyper parameter name and type separating them with comma. "
         "To pass several hyper parameters to script separate their specs with colons"
)
parser.add_argument(
    '-s',
    "--span",
    help="Range of hyper parameters values. Specify start, end, number of points and scale, separating them with commas"
         ". Ranges for  several hps separated by colons. Example:\n1e-5,1,20,log"
)
parser.add_argument(
    "-v",
    "--verbose",
    help="Regulate verbosity",
    action='store_true'
)
parser.add_argument(
    '-ng',
    '--no_git',
    help="don't add confs to git",
    action='store_true',
)

args = parser.parse_args()

base_path = helmo.util.path_help.get_path_from_path_rel_to_repo_root(os.path.join('helmo', 'experiments'))

confs = parse_path_comb(args.confs, filter_=False)

hp_names_with_types = args.hyper_parameters.split(':')
hp_names = list()
types = list()
for s in hp_names_with_types:
    spl = s.split(',')
    hp_names.append(spl[0])
    types.append(spl[1])

span = args.span.split(':')

points = list()
for string in span:
    points.append(
        get_points_from_range(string)
    )

indent = 4
for conf in confs:
    path_to_conf = os.path.join(base_path, conf)
    create_path(path_to_conf, file_name_is_in_path=True)
    tmpl = get_tmpl(hp_names)
    file_string = ''

    file_string += tmpl % tuple(hp_names) + '\n'
    file_string += tmpl % tuple(types) + '\n'
    for values in points:
        file_string += get_tmpl(values) % tuple(values) + '\n'
    file_string += '%s' % args.num_repeats
    with open(path_to_conf, 'w') as f:
        f.write(file_string)
    if args.verbose:
        print('\n' + ' '*indent + conf)
        print(file_string)

if not args.no_git:
    for conf in confs:
        command = 'git add %s' % conf
        os.system(command)
