import os
import sys


def check_cephfs():
    for p in sys.path:
        if "cephfs" in p:
            return True


def add_repo_2_sys_path(repo_name):
    sys_path = sys.path
    home = os.path.expanduser("~")
    if check_cephfs():
        home = os.path.join("cephfs", home)
    repo_path = os.path.join(home, repo_name)
    return [repo_path] + sys_path


def add_cephfs_to_path(path):
    if check_cephfs() and path[0] == '/' and path[1:7] != "cephfs":
        return "/cephfs" + path
    return path
