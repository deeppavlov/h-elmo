# from deeppavlov.core.commands.train import train_evaluate_model_from_config


import sys
from util.deal_with_cephfs import add_repo_2_sys_path
sys.path = add_repo_2_sys_path('DeepPavlov')




import one_str_lm_reader, one_str_lm_iterator, char_lm_vocab, lstm
from deeppavlov.core.commands.train import train_evaluate_model_from_config

train_evaluate_model_from_config('lstm_config.json')
