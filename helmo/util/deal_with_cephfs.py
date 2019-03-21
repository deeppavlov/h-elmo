import os


def check_cephfs():
    return "cephfs" in os.listdir('/')


def add_cephfs_to_path(path):
    if not isinstance(path, str):
        path = str(path)
    if check_cephfs() and path[0] == '/' and path[1:7] != "cephfs":
        return "/cephfs" + path
    return path
