import json
import argparse
import os

from helmo.util import interpreter
interpreter.extend_python_path_for_project()

from learning_to_learn.useful_functions import create_path, parse_path_comb, get_points_from_range, get_tmpl
from helmo.util.nested import nested_set
import helmo.util.path_help

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
    '-k',
    "--keys",
    help="Keys leading to entry where list has to be stored. Keys are separated with commas. Paths of keys leading to"
         "different entries are separated with colons. Example: key11,key12:key21,key22 will store lists in "
         "config[key11][key12] and config[key21][key22]."
)
parser.add_argument(
    '-t',
    '--types',
    help="Types of lists elements. Types of different lists are separated with colon."
)
parser.add_argument(
    '-s',
    "--span",
    help="Range of inserted list values. Specify start, end, number of points and scale, separating them with commas"
         ". Ranges for several hps separated by colons. Example:\n1e-5,1,20,log"
)
parser.add_argument(
    '-n',
    '--num_repeats',
    help="Number of times this conf will be repeated",
    default=1,
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

key_paths = [spl.split(',') for spl in args.keys.split(':')]
types = args.types.split(':')

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
    with open(path_to_conf) as f:
        config = json.load(f)
    for kp, values in zip(key_paths, points):
        nested_set(config, kp, values)
    with open(path_to_conf, 'w') as f:
        json.dump(config, f, indent=2)
    if not args.no_git:
        command = 'git add %s' % path_to_conf
        os.system(command)
