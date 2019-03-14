import sys
import os
sys.path += [
    os.path.join('/cephfs', os.path.expanduser('~/learning-to-learn')),
    os.path.expanduser('~/learning-to-learn'),
    os.path.join('/cephfs', os.path.expanduser('~/repos/learning-to-learn')),
    os.path.expanduser('~/repos/learning-to-learn'),
    '/cephfs/home/peganov/learning-to-learn',
    '/home/peganov/learning-to-learn',
]


def form_load_cmd(file_name, obj_name, imported_as):
    file_name.replace('/', '.')
    return "from helmo.nets.%s import %s as %s" % (file_name, obj_name, imported_as)