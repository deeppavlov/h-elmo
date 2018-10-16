import re


def hat_uppercase(text):
    return re.sub("\^([a-zа-яё])", lambda match: match.group(1).upper(), text)
