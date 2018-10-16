import sys
import os
sys.path += [
    os.path.join('/cephfs', os.path.expanduser('~/learning-to-learn')),
    os.path.expanduser('~/learning-to-learn'),
    os.path.join('/cephfs', os.path.expanduser('~/h-elmo')),
    os.path.expanduser('~/h-elmo'),
    os.path.join('/cephfs', os.path.expanduser('~/repos/learning-to-learn')),
    os.path.expanduser('~/repos/learning-to-learn'),
    os.path.join('/cephfs', os.path.expanduser('~/repos/h-elmo')),
    os.path.expanduser('~/repos/h-elmo'),
    '/cephfs/home/peganov/learning-to-learn',
    '/home/peganov/learning-to-learn',
    '/cephfs/home/peganov/h-elmo',
    '/home/peganov/h-elmo',
]
import argparse

from pathlib import Path  # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from learning_to_learn.useful_functions import get_missing_hp_sets, parse_path_comb
import helmo.util.organise as organise
parser = argparse.ArgumentParser()

parser.add_argument(
    "confs",
    help="Path to configs used for hp search. \nTo process several configs"
         " use following format '<path1>,<path2>,...<pathi>:<pathi+1>,..:..'.\nAll possible combinations of sets"
         " separated by colons (in specified order) will be processed. \nYou have to provide paths relative to "
         "script. Edge characters of <path> can't be '/'"
)
parser.add_argument(
    '-m',
    "--model",
    help="Optimized model type. It can be 'pupil' or optimizer or 'optimizer'. Default is 'optimizer'",
    default='optimizer',
)
parser.add_argument(
    "-v",
    "--verbose",
    help="Regulate verbosity",
    action='store_true'
)
args = parser.parse_args()

# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)

confs = parse_path_comb(args.confs)
eval_dirs = list()
for conf in confs:
    conf_name = '.'.join(os.path.split(conf)[-1].split('.')[:-1])
    base = os.path.join(organise.get_path_to_dir_with_results(conf), conf_name)
    dirs = list()
    for eval_file in os.listdir(base):
        if 'launch_log' not in eval_file:
            dirs.append(os.path.join(base, eval_file))
    eval_dirs.append(dirs)

for conf, dirs in zip(confs, eval_dirs):
    print(' '*8, conf)
    if len(dirs) == 0:
        print('no experiments found')
    for dir in dirs:
        print(' '*4, dir)
        missing_hp_sets, total_required = get_missing_hp_sets(conf, dir, args.model)
        num_missing = len(missing_hp_sets)
        print(('missing: %s' + ' '*10 + 'total_required: %s') % (num_missing, total_required))
        if args.verbose:
            for idx, hp_set in enumerate(missing_hp_sets):
                print(idx)
                for hp_name, hp_value in hp_set.items():
                    print(hp_name, hp_value)
                if idx < num_missing - 1:
                    print()

