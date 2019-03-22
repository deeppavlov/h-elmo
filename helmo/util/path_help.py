import os

def split_path_entirely(path):
    splitted_path = list()
    head, tail = os.path.split(path)
    while len(head) > 0 and len(tail) > 0:
        splitted_path = [tail] + splitted_path
        head, tail = os.path.split(head)
    splitted_path = [head] + splitted_path if len(head) > 0 else [tail] + splitted_path
    return splitted_path


def get_path_from_path_rel_to_repo_root(path):
    abspath_to_cwd = split_path_entirely(os.path.abspath(os.getcwd()))
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


def move_path_postfix_within_repo(
        *, path_to_smth_in_separator, separator="experiments", new_prefix_within_repo="expres"):
    rel_path = os.path.abspath(path_to_smth_in_separator).split('/' + separator + '/')[-1]
    results_dir = os.path.join(get_path_from_path_rel_to_repo_root(new_prefix_within_repo), rel_path)
    return results_dir


def prepend_restore_path_with_expres(restore_path):
    if isinstance(restore_path, str):
        return os.path.join(get_path_from_path_rel_to_repo_root('expres'), restore_path)
    restore_path = restore_path.copy()
    for k, v in restore_path.items():
        restore_path[k] = os.path.join(get_path_from_path_rel_to_repo_root('expres'), v)
    return restore_path


def get_save_path_from_config_path(config_path, directory_with_configs, results_directory_rel_to_repo_root):
    if directory_with_configs[0] == '/':
        directory_with_configs = directory_with_configs[1:]
    if directory_with_configs[-1] == '/':
        directory_with_configs = directory_with_configs[:-1]
    config_in_results = move_path_postfix_within_repo(
        path_to_smth_in_separator=config_path,
        separator=directory_with_configs,
        new_prefix_within_repo=results_directory_rel_to_repo_root
    )
    return os.path.splitext(config_in_results)[0]
