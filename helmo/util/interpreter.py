import os
import sys

from helmo.util.deal_with_cephfs import check_cephfs


def extend_python_path_for_project():
    append_dir_2_sys_path('~/learning-to-learn')
    append_dir_2_sys_path('~/h-elmo')
    append_dir_2_sys_path('~/repos/learning-to-learn')
    append_dir_2_sys_path('~/repos/h-elmo')


def prepend_repo_2_sys_path(repo_name):
    sys_path = sys.path
    home = os.path.expanduser("~")
    if check_cephfs():
        home = os.path.join("/cephfs", home.lstrip('/'))
    repo_path = os.path.join(home, repo_name)
    return [repo_path] + sys_path


def append_dir_2_sys_path(dir):
    dir = os.path.expanduser(dir)
    if check_cephfs():
        sys.path.append(os.path.join('/cephfs', dir.lstrip('/')))
    sys.path.append(dir)
