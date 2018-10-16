import re


def hat_uppercase(text):
    return re.sub("([A-ZА-ЯЁ])", lambda match: '^' + match.group(1).lower(), text)
