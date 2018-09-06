from deeppavlov.core.commands.train import train_evaluate_model_from_config
import one_str_lm_reader, one_str_lm_iterator, char_lm_vocab, lstm

train_evaluate_model_from_config('lstm_config.json')