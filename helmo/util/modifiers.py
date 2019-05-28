import os


def _read_pos_tags():
    tags = {}
    cwd = os.getcwd()
    os.chdir(os.path.split(__file__)[0])
    with open('../../pos_tags_meaning.txt') as f:
        for line in f:
            words = line.strip().split()
            tags[words[0]] = ' '.join(words[1:])
    os.chdir(cwd)
    return tags


def describe_pos_tag(tag):
    tags = _read_pos_tags()
    return tags[tag]
