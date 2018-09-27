# from deeppavlov.core.commands.train import train_evaluate_model_from_config


import sys
import os

sys.path += [os.path.join('/cephfs', os.path.expanduser('~/h-elmo')), os.path.expanduser('~/h-elmo')]

from helmo.util.deal_with_cephfs import add_repo_2_sys_path

sys.path = add_repo_2_sys_path('DeepPavlov')

from deeppavlov.core.commands.train import train_evaluate_model_from_config
import char_lm_vocab, one_str_lm_iterator, one_str_lm_reader, lstm

train_evaluate_model_from_config('lstm_config.json')
