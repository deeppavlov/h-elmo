import os
import argparse

import helmo.util.path_help

from learning_to_learn.useful_functions import get_missing_hp_sets, parse_path_comb

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
    save_path = os.path.splitext(
        helmo.util.path_help.move_path_postfix_within_repo(
            path_to_smth_in_separator=conf
        )
    )[0]
    dirs = list()
    for eval_file in os.listdir(save_path):
        if 'launch_log' not in eval_file:
            dirs.append(os.path.join(save_path, eval_file))
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

