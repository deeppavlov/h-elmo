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

from helmo.util.deal_with_cephfs import add_repo_2_sys_path

sys.path = add_repo_2_sys_path('DeepPavlov')

from deeppavlov.core.commands.train import train_evaluate_model_from_config
from helmo.deeppavlov_lm import char_lm_vocab, one_str_lm_iterator, one_str_lm_reader, lstm

train_evaluate_model_from_config('lstm_config.json')
