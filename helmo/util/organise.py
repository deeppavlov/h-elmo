import sys
import os
sys.path += [
    os.path.join('/cephfs', os.path.expanduser('~/learning-to-learn')),
    os.path.expanduser('~/learning-to-learn'),
    os.path.join('/cephfs', os.path.expanduser('~/h-elmo')),
    os.path.expanduser('~/h-elmo'),
    '/cephfs/home/peganov/learning-to-learn',
    '/home/peganov/learning-to-learn',
    '/cephfs/home/peganov/h-elmo',
    '/home/peganov/h-elmo',
]
from learning_to_learn.useful_functions import create_vocabulary


def full_path_split(path):
    splitted_path = list()
    head, tail = os.path.split(path)
    while len(head) > 0 and len(tail) > 0:
        splitted_path = [tail] + splitted_path
        head, tail = os.path.split(head)
    splitted_path = [head] + splitted_path if len(head) > 0 else [tail] + splitted_path
    return splitted_path


def get_path_from_path_rel_to_repo_root(path):
    abspath_to_cwd = full_path_split(os.path.abspath(os.getcwd()))
    # print("(organise.get_path_from_path_rel_to_repo_root)abspath_to_cwd:", abspath_to_cwd)
    if 'h-elmo' in abspath_to_cwd:
        cwd_relative_to_repo_root = abspath_to_cwd[abspath_to_cwd.index('h-elmo')+1:]
        if len(cwd_relative_to_repo_root) == 0:
            return path
        # print("(organise.get_path_from_path_rel_to_repo_root)cwd_relative_to_repo_root:", cwd_relative_to_repo_root)
        root_depth = len(cwd_relative_to_repo_root)
        return os.path.join(*['..'] * root_depth, path)
    elif 'h-elmo' in os.listdir(os.path.expanduser('~')):
        return os.path.join(os.path.expanduser('~/h-elmo'), path)
    else:
        if os.path.exists('/cephfs/home/peganov/h-elmo'):
            return os.path.join('/cephfs/home/peganov/h-elmo', path)
        elif os.path.exists('/home/peganov/h-elmo'):
            return os.path.join('/home/peganov/h-elmo', path)
        elif os.path.exists('/home/anton/h-elmo'):
            return os.path.join('/home/anton/h-elmo', path)
        else:
            return None


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


def _select(text, pointer, size):
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
    test_text, pointer = _select(text, 0, test_size)
    valid_text, pointer = _select(text, pointer, valid_size)
    train_text, _ = _select(text, pointer, train_size)
    if by_lines:
        test_text, valid_text, train_text = '\n'.join(test_text), '\n'.join(valid_text), '\n'.join(train_text)
    return test_text, valid_text, train_text


def get_path_to_dir_with_results(path_to_conf_or_script):
    rel_path = os.path.join(*os.path.split(os.path.abspath(path_to_conf_or_script).split('/experiments/')[-1])[:-1])
    results_dir = os.path.join(get_path_from_path_rel_to_repo_root('expres'), rel_path)
    return results_dir


def form_load_cmd(file_name, obj_name, imported_as):
    file_name.replace('/', '.')
    return "from helmo.nets.%s import %s as %s" % (file_name, obj_name, imported_as)


def load_vocabulary_with_unk(file_name):
    vocabulary = list()
    with open(file_name, 'r') as f:
        text = f.read()
        if '<UNK>' in text:
            vocabulary.append('<UNK>')
            text = text.replace('<UNK>', '')
        vocabulary += list(text)
    return vocabulary
