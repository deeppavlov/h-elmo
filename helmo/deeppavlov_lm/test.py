import sys
import os

from helmo.util import interpreter
interpreter.extend_python_path_for_project()

from helmo.util.interpreter import prepend_repo_2_sys_path

sys.path = prepend_repo_2_sys_path('DeepPavlov')

from deeppavlov.core.commands.train import train_evaluate_model_from_config
from helmo.deeppavlov_lm import char_lm_vocab, one_str_lm_iterator, one_str_lm_reader, lstm

train_evaluate_model_from_config('lstm_config.json')
