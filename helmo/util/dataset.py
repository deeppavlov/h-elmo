import os
import warnings

from helmo.util.path_help import get_path_from_path_rel_to_repo_root

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


class DatasetIndexError(Exception):
    def __init__(self, message):
        super().__init__(message)


def get_dataset_text(text_config):
    text = get_text(text_config['path'])
    text_length = len(text)
    start = int(text_config['start'])
    target_length = int(text_config['length'])
    if start < 0:
        start += text_length
    if start < 0 or start >= text_length:
        raise DatasetIndexError(
            "start of a dataset text is out of range\n\tstart = {}\n\ttext_length = {}".format(start, text_length)
        )
    if start + target_length > text_length:
        warnings.warn(
            "can not get dataset text of required size {}, dataset size is {} instead"
            "\n\tstart = {}\n\ttarget_length = {}\n\ttext_length = {}".format(
                target_length, text_length - start, start, target_length, text_length
            )
        )
    return text[start:start+target_length]


def get_datasets_using_config(dataset_config):
    if 'path' in dataset_config:
        text = get_text(dataset_config['path'])
        test_size = int(dataset_config['test_size'])
        valid_size = int(dataset_config['valid_size'])
        train_size = int(dataset_config['train_size']) if 'train_size' in dataset_config\
            else len(text) - test_size - valid_size
        test_text, valid_text, train_text = split_text(text, test_size, valid_size, train_size)
        return {'test': test_text}, {'valid': valid_text}, {'train': train_text}
    test_datasets = {}
    for dataset_name, text_config in dataset_config['test'].items():
        test_datasets[dataset_name] = get_dataset_text(text_config)
    valid_datasets = {}
    for dataset_name, text_config in dataset_config['valid'].items():
        valid_datasets[dataset_name] = get_dataset_text(text_config)
    train_dataset = dict(train=get_dataset_text(dataset_config['train']))
    return test_datasets, valid_datasets, train_dataset


def load_tag_descriptions(tag_file):
    with open(tag_file) as f:
        lines = f.readlines()
    descriptions = {}
    for line in filter(lambda x: len(x) > 0, lines):
        words = line.split()
        tag = words[0]
        description = ' '.join(words[1:])
        descriptions[tag] = description
    return descriptions
