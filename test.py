# from deeppavlov.core.commands.train import train_evaluate_model_from_config


# import importlib.util
# deeppavlov_spec = importlib.util.spec_from_file_location("deeppavlov", "/home/anton/DeepPavlov/deeppavlov/__init__.py")
# deeppavlov = importlib.util.module_from_spec(deeppavlov_spec)
# deeppavlov_spec.loader.exec_module(deeppavlov)
#
# train_spec = importlib.util.spec_from_file_location(
#     "deeppavlov", "/home/anton/DeepPavlov/deeppavlov/core/commands/train.py")
# train = importlib.util.module_from_spec(train_spec)
# train_spec.loader.exec_module(train)

import sys
from util.mod_sys_path import add_repo_2_sys_path
sys.path = add_repo_2_sys_path('DeepPavlov')

# sys.path.append('/home/anton/DeepPavlov')
# if '/home/anton/dpenv/src/deeppavlov' in sys.path:
#     sys.path.remove('/home/anton/dpenv/src/deeppavlov')



import one_str_lm_reader, one_str_lm_iterator, char_lm_vocab, lstm
from deeppavlov.core.commands.train import train_evaluate_model_from_config

train_evaluate_model_from_config('lstm_config.json')
