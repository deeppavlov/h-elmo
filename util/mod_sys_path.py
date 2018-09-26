import os
import sys

def add_repo_2_sys_path(repo_name):
    sys_path = sys.path
    cephfs_is_present = False
    for p in sys_path:
        if 'cephfs' in p:
            cephfs_is_present = True
    home = os.path.expanduser("~")
    if cephfs_is_present:
        home = os.path.join("cephfs", home)
    repo_path = os.path.join(home, repo_name)
    return [repo_path] + sys_path