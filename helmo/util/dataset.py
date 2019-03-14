import sys
import os

from helmo.util.path_help import get_path_from_path_rel_to_repo_root

sys.path += [
    os.path.join('/cephfs', os.path.expanduser('~/learning-to-learn')),
    os.path.expanduser('~/learning-to-learn'),
    os.path.join('/cephfs', os.path.expanduser('~/repos/learning-to-learn')),
    os.path.expanduser('~/repos/learning-to-learn'),
    '/cephfs/home/peganov/learning-to-learn',
    '/home/peganov/learning-to-learn',
]

from learning_to_learn.useful_functions import create_vocabulary
import helmo.util.path_help as path_help


def get_text(text_file_name):
    # print("(organise.get_text)path_rel_to_root('datasets'):", get_path_from_path_rel_to_repo_root('datasets'))
    # print("(organise.get_text)os.getcwd():", os.getcwd())
    dataset_file_path = os.path.join(get_path_from_path_rel_to_repo_root('datasets'), text_file_name)
    with open(dataset_file_path, 'r') as f:
        text = f.read()
    return text


def get_vocab_by_given_path(file_name, text, create=False):
    if os.path.isfile(file_name) and not create:
        vocabulary = load_vocabulary_with_unk(file_name)
    else:
        vocabulary = create_vocabulary(text, with_unk=True)
        if not os.path.exists(os.path.dirname(file_name)) and len(os.path.dirname(file_name)) > 0:
            os.makedirs(os.path.dirname(file_name))
        with open(file_name, 'w') as f:
            f.write(''.join(vocabulary))
    vocabulary_size = len(vocabulary)
    return vocabulary, vocabulary_size


def get_vocab(file_name, text, create=False):
    file_name = os.path.join(get_path_from_path_rel_to_repo_root('vocabs'), file_name)
    return get_vocab_by_given_path(file_name, text, create=create)


def _slice_shift(text, pointer, size):
    if isinstance(size, int):
        selection = text[pointer:pointer+size]
        pointer += size
    else:
        selection = text[size[0]:size[0] + size[1]]
        pointer = size[0] + size[1]
    return selection, pointer


def split_text(text, test_size, valid_size, train_size, by_lines=False):
    if by_lines:
        text = text.split('\n')
    test_text, pointer = _slice_shift(text, 0, test_size)
    valid_text, pointer = _slice_shift(text, pointer, valid_size)
    train_text, _ = _slice_shift(text, pointer, train_size)
    if by_lines:
        test_text, valid_text, train_text = '\n'.join(test_text), '\n'.join(valid_text), '\n'.join(train_text)
    return test_text, valid_text, train_text


def load_vocabulary_with_unk(file_name):
    vocabulary = list()
    with open(file_name, 'r') as f:
        text = f.read()
        if '<UNK>' in text:
            vocabulary.append('<UNK>')
            text = text.replace('<UNK>', '')
        vocabulary += list(text)
    return vocabulary