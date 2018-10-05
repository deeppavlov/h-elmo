import sys
import os
sys.path += [
    os.path.join('/cephfs', os.path.expanduser('~/learning-to-learn')),
    os.path.expanduser('~/learning-to-learn')
]
from learning_to_learn.useful_functions import create_vocabulary


def path_rel_to_root(path):
    abspath_to_current_file = os.path.abspath(os.getcwd())
    relative_to_repo_root = abspath_to_current_file.split('/h-elmo/')[-1]
    root_depth = len(relative_to_repo_root.split('/'))
    return os.path.join(*['..'] * root_depth, path)


def get_text(text_file_name):
    dataset_file_path = os.path.join(path_rel_to_root('datasets'), text_file_name)
    with open(dataset_file_path, 'r') as f:
        text = f.read()
    return text


def get_vocab(file_name, text, create=False):
    file_name = os.path.join(path_rel_to_root('vocabs'), file_name)
    if os.path.isfile(file_name) and not create:
        with open(file_name, 'r') as f:
            vocabulary = list(f.read())
    else:
        vocabulary = create_vocabulary(text)
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        with open(file_name, 'w') as f:
            f.write(''.join(vocabulary))
    vocabulary_size = len(vocabulary)
    return vocabulary, vocabulary_size


def _select(text, pointer, size):
    if isinstance(size, int):
        selection = text[:size]
        pointer = size
    else:
        selection = text[size[0]:size[0] + size[1]]
        pointer = size[0] + size[1]
    return selection, pointer


def split_text(text, test_size, valid_size, train_size):
    test_text, pointer = _select(text, 0, test_size)
    valid_text, pointer = _select(text, pointer, valid_size)
    train_text, _ = _select(text, pointer, train_size)
    return test_text, valid_text, train_text


def get_path_to_dir_with_results(path_to_conf_or_script):
    rel_path = os.path.join(*os.path.split(os.path.abspath(path_to_conf_or_script).split('/experiments/')[-1])[:-1])
    results_dir = os.path.join(path_rel_to_root('expres'), rel_path)
    return results_dir


